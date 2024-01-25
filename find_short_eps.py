import pandas as pd

df = pd.read_csv('dset_info.csv', index_col=0)
df = df.loc[df['usable'] & (df['split']=='test')]

show_names = set(x.split('-')[0] for x in df.index)

print(show_names)
all_shorts = []
for sn in show_names:
    df_sn = df.iloc[[sn in x for x in df.index]]
    five_shorts = df_sn.sort_values('duration_raw').iloc[:5]
    all_shorts.append(five_shorts)

shorts_df = pd.concat(all_shorts)
print(shorts_df)
shorts_df.to_csv('short_eps.csv')
breakpoint()
#selected_shorts = shorts_df.loc[['oltl-07-04-06','oltl-07-22-08','bb-12-17-15','bb-01-22-15','atwt-03-25-03','atwt-06-19-07','gl-08-23-05','gl-06-17-03','pc-11-02-01','pc-11-15-01']]
selected_shorts = shorts_df.loc[['gl-09-06-06','atwt-01-06-05','bb-01-02-15','pc-04-03-03']]
selected_shorts.to_csv('selected_short_eps.csv')

