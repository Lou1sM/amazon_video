import pandas as pd

df = pd.read_csv('dset_info.csv', index_col=0)
df = df.loc[df['usable'] & (df['split']!='nelly-excluded')]

show_names = set(x.split('-')[0] for x in df.index)

print(show_names)
all_shorts = []
for sn in show_names:
    df_sn = df.iloc[[sn in x for x in df.index]]
    five_shorts = df_sn.sort_values('duration').iloc[:5]
    all_shorts.append(five_shorts)

shorts_df = pd.concat(all_shorts)
print(shorts_df)
shorts_df.to_csv('short_eps.csv')
breakpoint()

