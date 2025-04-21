import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from PIL import Image
from scipy.spatial import cKDTree
from skimage.transform import resize
from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_directories(output_base_path):
    """
    Create standard output directories under a given base path.

    Returns:
        qc_output_dir, density_dir, segmentation_dir, storyboard_dir
    """
    qc = os.path.join(output_base_path, "QC")
    dens = os.path.join(qc, "density_maps")
    seg = os.path.join(qc, "segmentation")
    sb = os.path.join(qc, "storyboards")
    for d in (qc, dens, seg, sb):
        os.makedirs(d, exist_ok=True)
    return qc, dens, seg, sb


def compute_density_map(mask, cells, window_size, stride):
    h, w = mask.shape
    dm = np.zeros((h, w), float)
    ids = set(int(c.split('_')[-1]) for c in cells)
    if not ids.intersection(np.unique(mask)):
        return dm
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            block = mask[y:y+window_size, x:x+window_size]
            cnt = len(set(np.unique(block)).intersection(ids))
            if cnt:
                dm[y:y+window_size, x:x+window_size] += cnt
    return dm


def perform_density_filtering(
    adata,
    qc_dir,
    density_dir,
    window_size=50,
    stride=10,
    density_threshold=550,
    region_threshold=0.8,
    density_scale=(0,800)
):
    """
    Spatial density QC per FOV with progress bar.
    """
    adata.obs['QC_filter_low_quality_region'] = False
    low = set(adata.obs.index[adata.obs['QC_low_intensity_filter']])
    fovs = sorted(adata.obs['fov'].unique())
    region_map = {}

    for fov in tqdm(fovs, desc="🔍 Density QC", unit="FOV"):
        seg = adata.uns.get('spatial', {}).get(fov, {}).get('segmentation')
        if seg is None:
            logger.warning(f"Skipping {fov}, no segmentation.")
            continue
        filtered = [c for c in low if c.startswith(f"{fov}_")]
        dm = compute_density_map(seg, filtered, window_size, stride)
        # save
        plt.figure(figsize=(6,6)); plt.imshow(dm, cmap='hot', vmin=density_scale[0], vmax=density_scale[1]); plt.axis('off')
        plt.savefig(os.path.join(density_dir, f"{fov}.png"), dpi=200, bbox_inches='tight'); plt.close()
        # determine region cells
        m = dm > density_threshold
        if m.shape != seg.shape:
            m = resize(m.astype(float), seg.shape) > 0.5
        cnt, rg = defaultdict(int), defaultdict(int)
        for y in range(seg.shape[0]):
            for x in range(seg.shape[1]):
                lab = seg[y,x]
                if lab:
                    cnt[lab] += 1
                    if m[y,x]: rg[lab] += 1
        keep = {f"{fov}_{lab}" for lab, c in rg.items() if c/cnt[lab] > region_threshold}
        region_map[fov] = keep
        mask = adata.obs['fov']==fov
        adata.obs.loc[mask, 'QC_filter_low_quality_region'] = adata.obs.loc[mask].index.isin(keep)
    return adata, region_map


def perform_dr_filtering(adata, radius, good_frac=0.75):
    """
    Generic DR-based neighborhood filtering.
    """
    # find DR key
    dr_key = next((k for k in adata.obsm if k.startswith('X_') and k!='X_spatial'), None)
    if dr_key is None:
        logger.error("No DR embedding found, skipping.")
        return adata
    logger.info(f"🔧 Using DR embedding: {dr_key}")
    pts = adata.obsm[dr_key]
    logger.debug(f"DR filter on {pts.shape[0]} cells with radius={radius}")
    good = adata.obs['filtering_status']=='Unfiltered'
    tree_all = cKDTree(pts)
    pts_good = pts[good.values]
    if pts_good.size==0:
        adata.obs['QC_fraction_filtered'] = 1.
        adata.obs['QC_dr_based_filter'] = False
        return adata
    tree_good = cKDTree(pts_good)
    # neighbors
    tot = np.array([len(n) for n in tree_all.query_ball_point(pts, r=radius)])
    ok  = np.array([len(n) for n in tree_good.query_ball_point(pts, r=radius)])
    frac = np.zeros_like(tot, float)
    valid = tot>0
    frac[valid] = ok[valid]/tot[valid]
    adata.obs['QC_fraction_filtered'] = frac
    flag = frac<good_frac
    adata.obs['QC_dr_based_filter'] = flag
    msg = int(flag.sum())
    logger.info(f"🎯 DR filtering flagged {msg}/{len(frac)} bad DR cells (frac < {good_frac})")
    idx = flag & (adata.obs['filtering_status']=='Unfiltered')
    adata.obs.loc[idx,'filtering_status']='bad dr cluster'
    return adata


def generate_segmentation_images(adata, fovs, density_dir, segmentation_dir, region_map):
    os.makedirs(segmentation_dir, exist_ok=True)
    for fov in tqdm(fovs, desc="🎨 Segmentation QC", unit="FOV"):
        seg = adata.uns.get('spatial',{}).get(fov,{}).get('segmentation')
        if seg is None: continue
        H,W=seg.shape
        low = set(adata.obs.query("fov==@fov and QC_low_intensity_filter").index)
        reg = region_map.get(fov, set())
        drf = set(adata.obs.query("fov==@fov and QC_dr_based_filter").index)
        img = np.full((H,W,3), 255, np.uint8)
        for y in range(H):
            for x in range(W):
                lab=seg[y,x]
                if not lab: continue
                key=f"{fov}_{lab}"
                if key in low: img[y,x]=(255,0,0)
                elif key in reg: img[y,x]=(0,255,0)
                elif key in drf: img[y,x]=(255,255,0)
        plt.imsave(os.path.join(segmentation_dir,f"{fov}.png"), img)
    logger.info("Segmentation overlays done")


def create_storyboard(paths, out, cols=10):
    if not paths: return
    imgs=[Image.open(p) for p in paths]
    w,h=min(i.width for i in imgs), min(i.height for i in imgs)
    board=Image.new('RGB',(cols*w, ((len(imgs)+cols-1)//cols)*h),(255,255,255))
    for i,im in enumerate(imgs): board.paste(im.resize((w,h)),((i%cols)*w,(i//cols)*h))
    board.save(out)


def generate_storyboards(qc_dir, dens_dir, seg_dir, sb_dir, fovs):
    segs=[os.path.join(seg_dir,f"{f}.png") for f in fovs]
    dens=[os.path.join(dens_dir,f"{f}.png") for f in fovs]
    if segs: create_storyboard(segs,os.path.join(sb_dir,'segmentation_storyboard.png'))
    if dens: create_storyboard(dens,os.path.join(sb_dir,'density_storyboard.png'))


def generate_dr_plot(adata, qc_dir):
    dr_key = next((k for k in adata.obsm if k.startswith('X_') and k!='X_spatial'),None)
    if dr_key is None: return
    coords=adata.obsm[dr_key]
    status=adata.obs['filtering_status']
    colors=np.full(len(coords), 'lightgray', object)
    colors[status=='low intensity cell']='red'
    colors[status=='low quality region']='green'
    colors[status=='bad dr cluster']='yellow'
    fig,ax=plt.subplots(figsize=(6,5))
    ax.scatter(coords[:,0],coords[:,1],c=colors, s=1, alpha=0.6)
    ax.set_title(f"Filtering Results after Dimension Reduction ({dr_key})")
    ax.set_xlabel(f"{dr_key} 1")
    ax.set_ylabel(f"{dr_key} 2")
    ax.legend(
        handles=[
            plt.Line2D([], [], marker='o', color='w', markerfacecolor=c, label=l, markersize=6)
            for l, c in [
                ('Low Intensity Cell', 'red'),
                ('Low Quality Region', 'green'),
                ('Bad DR Cluster', 'yellow'),
                ('Unfiltered', 'lightgray')
            ]
        ],
        title='Filter',
        loc='upper right',
        markerscale=0.5,  # Shrinks marker size
        fontsize='x-small',  # Shrinks label text
        title_fontsize='small'  # Shrinks title
    )
    plt.savefig(os.path.join(qc_dir,'dr_qc.png'),dpi=200,bbox_inches='tight')
    plt.close()


def generate_summary_tables(adata, qc_dir):
    total=len(adata.obs)
    status=adata.obs['filtering_status']
    steps=[('Total',None),('Low Intensity','low intensity cell'),('Low Quality','low quality region'),('Bad DR','bad dr cluster')]
    rows=[{'Filter Step':n,'Count':(status==v).sum() if v else total,'Percent':((status==v).sum()/total*100) if v else 100}
          for n,v in steps]
    pd.DataFrame(rows).to_csv(os.path.join(qc_dir,'overall_stats.csv'),index=False)
    fov_list=adata.obs['fov'].unique()
    fov_rows=[]
    for fov in fov_list:
        sub=status[adata.obs['fov']==fov]
        d={'FOV':fov}
        for n,v in steps: d[n]=(sub==v).sum() if v else len(sub)
        fov_rows.append(d)
    pd.DataFrame(fov_rows).to_csv(os.path.join(qc_dir,'per_fov_stats.csv'),index=False)


def run_qc_from_memory(args, adata):
    qc_dir,dens,seg,sb=create_directories(args.output_base_path)
    logger.debug('🚀 Running QC pipeline')
    logger.info(f'🔧 Low-intensity threshold={args.low_intensity_threshold}')
    logger.info(f'🔧 Density window={args.qc_window_size},stride={args.qc_stride}')
    logger.info(f'🔧 Density thresh={args.qc_density_threshold},region={args.qc_region_threshold}')
    logger.info(f'🔧 DR radius={args.radius_DRfilter}')

    # intensity filter
    tot=adata.layers['sum_unhuddle'].sum(axis=1)
    adata.obs['total_intensity']=tot
    adata.obs['QC_low_intensity_filter']=tot<args.low_intensity_threshold
    low_count=int((adata.obs['QC_low_intensity_filter']).sum())

    # density
    logger.debug('🔍 Density QC')
    adata,region_map=perform_density_filtering(
        adata,qc_dir,dens,
        window_size=args.qc_window_size,stride=args.qc_stride,
        density_threshold=args.qc_density_threshold,region_threshold=args.qc_region_threshold,
        density_scale=tuple(args.qc_plot_density_scale)
    )
    region_count=int((adata.obs['QC_filter_low_quality_region']).sum())
    logger.info(f'🔧 Post-density filter counts: low_intensity={low_count}, low_quality_region={region_count}')

    # status init
    adata.obs['filtering_status']='Unfiltered'
    adata.obs.loc[adata.obs['QC_filter_low_quality_region'],'filtering_status']='low quality region'
    adata.obs.loc[adata.obs['QC_low_intensity_filter'],'filtering_status']='low intensity cell'

    # DR filter
    logger.info('🎲 DR filtering')
    adata=perform_dr_filtering(adata,args.radius_DRfilter)
    dr_bad=int((adata.obs['QC_dr_based_filter']).sum())
    logger.info(f'🔧 Post-DR filter count: bad_dr_cluster={dr_bad}')

    # segmentation
    logger.info('🎨 Segmentation overlays')
    generate_segmentation_images(adata,list(region_map),dens,seg,region_map)

    # storyboards
    logger.info('📚 Storyboards')
    generate_storyboards(qc_dir,dens,seg,sb,list(region_map))

    # DR plot
    logger.info('📈 DR plot')
    generate_dr_plot(adata,qc_dir)

    # summary
    logger.info('📊 Summaries')
    generate_summary_tables(adata,qc_dir)

    # final save
    logger.info('💾 Save filtered')
    keep=~(adata.obs['QC_low_intensity_filter']|adata.obs['QC_filter_low_quality_region'])
    kept_count=int(keep.sum())
    logger.info(f'🔧 Final cells retained={kept_count}/{adata.n_obs}')
    adata=adata[keep].copy()
    path=os.path.join(args.output_base_path,'adata_objects','adata1.h5ad')
    adata.write_h5ad(path)
    logger.info(f'💾 Saved QC-adata: {path}')
    print(f'✅ Saved QC-adata: {path}')
    del adata