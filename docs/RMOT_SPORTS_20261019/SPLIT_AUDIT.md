# Refer-Sports split audit

Status: `VIDEO_DISJOINT_SPLITS_VERIFIED`

The protocol uses three explicitly separated roles:

| role | videos | expressions | use |
| --- | ---: | ---: | --- |
| full Sports train | 22 | 181 | full-budget YesTrack run and original baseline route |
| internal selection train core | 17 | 136 | checkpoint/temporal training |
| internal selection validation | 5 | 45 | checkpoint and threshold selection |
| historical test | 19 | 142 | final offline TrackEval only |

The five internal-validation videos are fixed and video-disjoint from the 17
training-core videos. The 19 historical test videos are disjoint from both.
The exact lists are versioned under `configs/internal_split_20261019/` and
`configs/refer_sports_eval_sequences_20261019.txt`.

Test GT is never used for training, candidate generation, checkpoint choice,
threshold tuning, or tracker input. The formal evaluator may use a `val`
directory name for loader compatibility; that directory is the historical
test input layout, not the five-video internal selection split.
