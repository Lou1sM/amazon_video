import pandas as pd
from utils import metric_names


for dset_name in ['osvd', 'bbc-max', 'bbc-mean']:
    main_results = pd.read_csv(f'segmentation-results/{dset_name}/ours-unifs.csv', index_col=0)
    clustering_baseline_results = pd.read_csv(f'segmentation-results/{dset_name}/baselines.csv', index_col=0)
    lgss_results = pd.read_csv(f'segmentation-results/{dset_name}/lgss.csv', index_col=0)
    lgss_results.columns = ['lgss']
    combined = pd.concat([main_results, lgss_results.T, clustering_baseline_results], axis=0)
    combined = combined[metric_names]
    print('\n',dset_name.upper())
    print(combined)
    combined.to_csv(f'segmentation-results/{dset_name}/combined.csv')
