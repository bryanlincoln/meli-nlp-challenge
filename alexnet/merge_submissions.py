import pandas as pd

sub_pt = pd.read_csv('submission_pt.csv')
sub_es = pd.read_csv('submission_es.csv')

sub = pd.concat([sub_pt, sub_es], axis=0)
sub = sub.sort_values('id')
sub.to_csv('submission.csv', index=False)

print(sub[:20])
