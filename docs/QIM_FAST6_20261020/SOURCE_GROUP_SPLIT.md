# Historical source-group split

The archived manifest is:

`/data1/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/metadata/datasets/refer_sports_manifest_global_ids_rev2.json`

Manifest SHA256:

`160245526fae5c7b5c043e5fce4ec5ddb90bed1bd0da4e4f2a8fcf8b2354da`

The recovered protocol contains 22 historical train videos / 181 train
expressions and 19 historical test videos / 142 test expressions. The
historical 141-compatible protocol excludes only:

`v_4LXTUim5anY_c010:players`

The 19 historical test videos are physically stored under
`SportsMOT/sportsmot_publish/dataset/train`, together with their global-ID
labels. This physical directory is an artifact of the archived Refer-Sports
protocol; it does not mean that the public 150-video SportsMOT `dataset/test`
split is being used. The evaluator locks video and expression selection to the
archived manifest.

The final report must distinguish QIM group-heldout validation from a fully
source-independent validation because the shared non-QIM initialization may
have seen related sources during its historical training.
