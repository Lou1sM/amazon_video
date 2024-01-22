import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


soaps = ['Doctors', 'Eastenders', 'Casualty', 'Holby City',]
comedies = ['Dad\'s Army', 'Two Pints of Lager and a Packet of Crisps', '\'Allo \'Allo!', 'Hi-de-Hi!']
def hms_to_float_mins(s):
    h,m,s = s.split(':')
    return float(h)*60 + float(m) + float(s)/60

data = pd.read_csv('bbc_data.csv',header=0)
data = data.loc[~data['Diskref'].isna()]
data = data.fillna('')

ts_, cs = np.unique(data['EIT_duration'],return_counts=True)
ts = [hms_to_float_mins(x) for x in ts_]
fig, axes = plt.subplots(2,1)
axes[0].bar(ts, cs, log=True, width=5.0)
plt.xticks(np.arange(0,int(max(ts)),30))
axes[0].set(ylabel='Count (log)')
axes[0].title.set_text('Distribution of Video Lengths')

axes[1].bar(ts, cs, width=5.0)
axes[1].set(xlabel='Duration (mins)', ylabel='Count (absolute)')

fig.tight_layout()
plt.savefig('BBC_video_length_dist.png')

unique_shows, counts = np.unique(data['PIPs_title'],return_counts=True)
print('Num different shows:', len(unique_shows))
maybe_films = np.array([x for x,c in zip(unique_shows,counts) if c==1])
print(f'Num appearing only once (not all films): {maybe_films.shape[0]}')
print('Examples:')
for fn in np.random.choice(maybe_films,size=10,replace=False):
    print('\t',fn)

shows_by_count = sorted(zip(unique_shows,counts), key=lambda x:x[1], reverse=True)
print('\nTen most common shows:')
for showcount in shows_by_count[:100]:
    argmx,mx = showcount
    argmx += ':'+' '*(25-len(argmx))
    print(f'{argmx} {mx} occurrences')


long_summ_lens = np.array([len(x.split()) for x in data['PIPs_synopsis_long']])
mid_summ_lens = np.array([len(x.split()) for x in data['PIPs_synopsis_mid']])
short_summ_lens = np.array([len(x.split()) for x in data['PIPs_synopsis_short']])

print(f'\nShort synopsis length: {short_summ_lens.mean():.1f} words +/-({short_summ_lens.std():.2f})')
print(f'Med synopsis length: {mid_summ_lens.mean():.1f} words +/-({mid_summ_lens.std():.2f})')
print(f'Long synopsis length: {long_summ_lens.mean():.1f} words +/-({long_summ_lens.std():.2f})')

def get_and_save_by_type(type_list,name):
    df = data.loc[data['PIPs_title'].isin(type_list)]
    df = df.loc[df['PIPs_colour']=='colour'][['Diskref', 'PIPs_title', 'EIT_start_datetime', 'EIT_title']]
    df.to_csv(f'to_give_amazon_{name}.csv')
    return df

soaps_df = get_and_save_by_type(soaps, 'soaps')
comedies_df = get_and_save_by_type(comedies, 'comedies')
both_df = get_and_save_by_type(soaps+comedies, 'both')

