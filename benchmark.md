Comparing word per second rate using different loss function on different configuration.
loss_fct: {softmax, sampledsoftmax, nce, seq, seqex}
config: {smsm, smlg, lgsm, lglg}


23 nov, 2016 - pltrdy
Using commit: 571e7ad711b975901c5894277e89fe0b511a1de0

Configs are called '(sm|lg)(sm|lg)'
  First part stands for the default config as defined in 'config.py'.
  Second stands for vocabulary size: 'sm'= 10K; 'lg'=150K
    ex: 'smsm': {hidden_size: 200; num_steps: 20; vocab_size: 10000}
        'lglg': {hidden_size: 650; num_steps: 35; vocab_size: 150000}

All cases are train for 1 epoch. Not saving checkpoint files.
Using GPU nVidia 980Ti.

k-Word per sec:
  
  | config | softmax| sampledsoftmax | nce    | seq   | seqex |
  |--------|--------|----------------|--------|-------|-------|
  | smsm   | 18.5   | 19. 0          | 19.3   | 18.6  | 1.8   |
  | smlg   | 1.5    | 1.3            | 1.3    | 1.3   | 1.4   |
  | lgsm   | 8.8    | 8.8            | 8.9    | 8.8   | 8.8   |
  | lglg   | 1.0    | 1.0            | 1.0    | 0.93  | 9.9   |

Run time:
  | config | softmax| sampledsoftmax | nce    | seq   | seqex |
  |--------|--------|----------------|--------|-------|-------|
  | smsm   | 0m56s  | 0m55s          | 0m54s  | 0m56s | 0m56s |
  | smlg   | 10m45s | 12m28s         | 12m14s | 12m53s| 12m6s |
  | lgsm   | 1m55s  | 1m55s          | 1m55s  | 1m56s | 1m55s |
  | lglg   | 16m27s | 16m7s          | 16m31s | 17m48s| 16m42s|


======== softmax smsm ============
Starting training from epoch 1 using softmax
Epoch: 1 Learning rate: 1.000
0.000 perplexity: 2.718 speed: 2134 wps
0.200 perplexity: 2.718 speed: 18319 wps
0.400 perplexity: 2.718 speed: 18595 wps
0.600 perplexity: 2.718 speed: 18452 wps
0.799 perplexity: 2.718 speed: 18519 wps
0.999 perplexity: 2.718 speed: 18589 wps
Epoch: 1 Train Perplexity: 2.718
Epoch: 1 Valid Perplexity: 2.718

real	0m56.386s
user	0m36.904s
sys	0m1.999s
======== softmax smlg ============
Starting training from epoch 1 using softmax
Epoch: 1 Learning rate: 1.000
0.000 perplexity: 2.719 speed: 589 wps
0.200 perplexity: 2.719 speed: 1559 wps
0.400 perplexity: 2.719 speed: 1557 wps
0.600 perplexity: 2.719 speed: 1561 wps
0.799 perplexity: 2.719 speed: 1559 wps
0.999 perplexity: 2.719 speed: 1558 wps
Epoch: 1 Train Perplexity: 2.719
Epoch: 1 Valid Perplexity: 2.719

real	10m45.269s
user	1m21.774s
sys	7m24.425s
======== softmax lgsm ============
Starting training from epoch 1 using softmax
Epoch: 1 Learning rate: 1.000
0.001 perplexity: 2.718 speed: 2021 wps
0.200 perplexity: 2.718 speed: 8696 wps
0.400 perplexity: 2.718 speed: 8794 wps
0.600 perplexity: 2.718 speed: 8831 wps
0.800 perplexity: 2.718 speed: 8847 wps
0.999 perplexity: 2.718 speed: 8856 wps
Epoch: 1 Train Perplexity: 2.718
Epoch: 1 Valid Perplexity: 2.718

real	1m55.477s
user	1m14.396s
sys	0m13.901s
======== softmax lglg ============
Starting training from epoch 1 using softmax
Epoch: 1 Learning rate: 1.000
0.001 perplexity: 2.718 speed: 499 wps
0.200 perplexity: 2.718 speed: 998 wps
0.400 perplexity: 2.718 speed: 1002 wps
0.600 perplexity: 2.718 speed: 1003 wps
0.800 perplexity: 2.718 speed: 1003 wps
0.999 perplexity: 2.718 speed: 1004 wps
Epoch: 1 Train Perplexity: 2.718
Epoch: 1 Valid Perplexity: 2.718

real	16m27.892s
user	5m4.984s
sys	10m57.995s
======== sampledsoftmax smsm ============
Starting training from epoch 1 using sampledsoftmax
Epoch: 1 Learning rate: 1.000
0.000 perplexity: 433.816 speed: 2247 wps
0.200 perplexity: 65.354 speed: 18576 wps
0.400 perplexity: 46.238 speed: 18883 wps
0.600 perplexity: 37.804 speed: 18974 wps
0.799 perplexity: 32.942 speed: 19029 wps
0.999 perplexity: 29.426 speed: 19065 wps
Epoch: 1 Train Perplexity: 29.423
Epoch: 1 Valid Perplexity: 2.718

real	0m55.231s
user	0m39.916s
sys	0m2.230s
======== sampledsoftmax smlg ============
Starting training from epoch 1 using sampledsoftmax
Epoch: 1 Learning rate: 1.000
0.000 perplexity: 7763.354 speed: 882 wps
0.200 perplexity: 195.498 speed: 1328 wps
0.400 perplexity: 112.621 speed: 1330 wps
0.600 perplexity: 80.560 speed: 1327 wps
0.799 perplexity: 63.625 speed: 1324 wps
0.999 perplexity: 52.725 speed: 1324 wps
Epoch: 1 Train Perplexity: 52.712
Epoch: 1 Valid Perplexity: 2.718

real	12m28.659s
user	1m21.796s
sys	8m11.847s
======== sampledsoftmax lgsm ============
Starting training from epoch 1 using sampledsoftmax
Epoch: 1 Learning rate: 1.000
0.001 perplexity: 527.331 speed: 2216 wps
0.200 perplexity: 91.520 speed: 8729 wps
0.400 perplexity: 65.226 speed: 8823 wps
0.600 perplexity: 53.131 speed: 8853 wps
0.800 perplexity: 45.690 speed: 8871 wps
0.999 perplexity: 40.416 speed: 8879 wps
Epoch: 1 Train Perplexity: 40.416
Epoch: 1 Valid Perplexity: 2.718

real	1m55.511s
user	0m58.666s
sys	0m7.412s
======== sampledsoftmax lglg ============
Starting training from epoch 1 using sampledsoftmax
Epoch: 1 Learning rate: 1.000
0.001 perplexity: 6112.823 speed: 762 wps
0.200 perplexity: 331.085 speed: 1038 wps
0.400 perplexity: 191.248 speed: 1036 wps
0.600 perplexity: 136.932 speed: 1033 wps
0.800 perplexity: 107.444 speed: 1034 wps
0.999 perplexity: 88.283 speed: 1033 wps
Epoch: 1 Train Perplexity: 88.283
Epoch: 1 Valid Perplexity: 2.718

real	16m7.157s
user	3m31.504s
sys	11m21.044s
======== nce smsm ============
Starting training from epoch 1 using nce
Epoch: 1 Learning rate: 1.000
0.000 perplexity: 3676184236358061129886776616201229966343348248202056557874255830468034736261359458377007104.000 speed: 2550 wps
0.200 perplexity: 77809217664336831652568805182275584.000 speed: 18962 wps
0.400 perplexity: 3016392715932537450397696.000 speed: 19248 wps
0.600 perplexity: 20632851058366812160.000 speed: 19350 wps
0.799 perplexity: 13102158493650394.000 speed: 19399 wps
0.999 perplexity: 81253772631253.984 speed: 19427 wps
Epoch: 1 Train Perplexity: 80620958406476.625
Epoch: 1 Valid Perplexity: 2.718

real	0m54.372s
user	0m39.012s
sys	0m2.293s
======== nce smlg ============
Starting training from epoch 1 using nce
Epoch: 1 Learning rate: 1.000
0.000 perplexity: 95789555133295476482829315177948066713979035469407105287843189931903234142357718132007036161852557828044092624863089127169648042231979835392.000 speed: 862 wps
0.200 perplexity: 30582413976711689958274457636490510358993938906309397404106756525934192811921919521394262016.000 speed: 1366 wps
0.400 perplexity: 558713477179748476413234960272580109687835687226914765938589459173316378522288128.000 speed: 1371 wps
0.600 perplexity: 97497514254947662107786859916427615687130349529113168201259809627300691968.000 speed: 1372 wps
0.799 perplexity: 645436870473306407388096564799857812549168659375514636183611943944192.000 speed: 1369 wps
0.999 perplexity: 67101975936713950391543417074453281808344069047329434782364860416.000 speed: 1368 wps
Epoch: 1 Train Perplexity: 66257601557878513224102384545068903447629954306615302842512048128.000
Epoch: 1 Valid Perplexity: 2.718

real	12m14.447s
user	1m27.259s
sys	8m52.528s
======== nce lgsm ============
Starting training from epoch 1 using nce
Epoch: 1 Learning rate: 1.000
0.001 perplexity: 11644768214856983674079869189929390550576967452314806753912363676224661443323436824772080566272.000 speed: 2114 wps
0.200 perplexity: 184434147091578482928771748953106631505215488.000 speed: 8759 wps
0.400 perplexity: 602028972630751251863216909713408.000 speed: 8863 wps
0.600 perplexity: 641999219117980930187198464.000 speed: 8893 wps
0.800 perplexity: 67498553559542229106688.000 speed: 8907 wps
0.999 perplexity: 102700700934640779264.000 speed: 8918 wps
Epoch: 1 Train Perplexity: 102700700934640779264.000
Epoch: 1 Valid Perplexity: 2.718

real	1m55.021s
user	1m15.359s
sys	0m12.840s
======== nce lglg ============
Starting training from epoch 1 using nce
Epoch: 1 Learning rate: 1.000
0.001 perplexity: 73937349358642363592451208833541755354123316614045300600599539591165208707428208194894593969721851963440264747519474454732694554989939392512.000 speed: 775 wps
0.200 perplexity: 532996660979092683229091191105928940878011339498208324346107292371420082394924910674515647405228032.000 speed: 1021 wps
0.400 perplexity: 96673338920984179680933441432101021998986108505766244449845908292360151690345647425191936.000 speed: 1017 wps
0.600 perplexity: 92746946215174658251436878875386826174543419089994447098579846353291067884000247808.000 speed: 1013 wps
0.800 perplexity: 1108742960110222206440956538176031074918735038448900127323727902237016373329920.000 speed: 1012 wps
0.999 perplexity: 223290659795969426443684184253213250975841353525374220003077096640700481536.000 speed: 1011 wps
Epoch: 1 Train Perplexity: 223290659795969426443684184253213250975841353525374220003077096640700481536.000
Epoch: 1 Valid Perplexity: 2.718

real	16m31.939s
user	3m49.038s
sys	12m3.585s
======== seq smsm ============
Starting training from epoch 1 using seq
Epoch: 1 Learning rate: 1.000
0.000 perplexity: 1.023 speed: 2752 wps
0.200 perplexity: 1.022 speed: 18331 wps
0.400 perplexity: 1.020 speed: 18533 wps
0.600 perplexity: 1.019 speed: 18606 wps
0.799 perplexity: 1.019 speed: 18635 wps
0.999 perplexity: 1.018 speed: 18667 wps
Epoch: 1 Train Perplexity: 1.018
Epoch: 1 Valid Perplexity: 2.718

real	0m56.019s
user	0m37.023s
sys	0m1.893s
======== seq smlg ============
Starting training from epoch 1 using seq
Epoch: 1 Learning rate: 1.000
0.000 perplexity: 1.030 speed: 943 wps
0.200 perplexity: 1.029 speed: 1336 wps
0.400 perplexity: 1.025 speed: 1323 wps
0.600 perplexity: 1.023 speed: 1310 wps
0.799 perplexity: 1.022 speed: 1304 wps
0.999 perplexity: 1.022 speed: 1299 wps
Epoch: 1 Train Perplexity: 1.022
Epoch: 1 Valid Perplexity: 2.718

real	12m53.772s
user	1m14.733s
sys	9m27.024s
======== seq lgsm ============
Starting training from epoch 1 using seq
Epoch: 1 Learning rate: 1.000
0.001 perplexity: 1.013 speed: 2329 wps
0.200 perplexity: 1.013 speed: 8664 wps
0.400 perplexity: 1.012 speed: 8749 wps
0.600 perplexity: 1.011 speed: 8778 wps
0.800 perplexity: 1.011 speed: 8795 wps
0.999 perplexity: 1.011 speed: 8804 wps
Epoch: 1 Train Perplexity: 1.011
Epoch: 1 Valid Perplexity: 2.718

real	1m56.511s
user	1m13.862s
sys	0m15.706s
======== seq lglg ============
Starting training from epoch 1 using seq
Epoch: 1 Learning rate: 1.000
0.001 perplexity: 1.017 speed: 707 wps
0.200 perplexity: 1.017 speed: 937 wps
0.400 perplexity: 1.016 speed: 936 wps
0.600 perplexity: 1.014 speed: 934 wps
0.800 perplexity: 1.014 speed: 933 wps
0.999 perplexity: 1.013 speed: 930 wps
Epoch: 1 Train Perplexity: 1.013
Epoch: 1 Valid Perplexity: 2.718

real	17m48.972s
user	4m57.214s
sys	12m25.372s
======== seqex smsm ============
Starting training from epoch 1 using seqex
Epoch: 1 Learning rate: 1.000
0.000 perplexity: 8522.650 speed: 2752 wps
0.200 perplexity: 644.668 speed: 18312 wps
0.400 perplexity: 442.684 speed: 18567 wps
0.600 perplexity: 356.128 speed: 18651 wps
0.799 perplexity: 307.309 speed: 18686 wps
0.999 perplexity: 271.955 speed: 18711 wps
Epoch: 1 Train Perplexity: 271.925
Epoch: 1 Valid Perplexity: 2.718

real	0m56.057s
user	0m36.644s
sys	0m1.930s
======== seqex smlg ============
Starting training from epoch 1 using seqex
Epoch: 1 Learning rate: 1.000
0.000 perplexity: 128567.160 speed: 1240 wps
0.200 perplexity: 2059.935 speed: 1384 wps
0.400 perplexity: 1126.417 speed: 1383 wps
0.600 perplexity: 788.659 speed: 1382 wps
0.799 perplexity: 617.880 speed: 1380 wps
0.999 perplexity: 508.545 speed: 1381 wps
Epoch: 1 Train Perplexity: 508.419
Epoch: 1 Valid Perplexity: 2.718

real	12m6.647s
user	1m15.478s
sys	8m40.367s
======== seqex lgsm ============
Starting training from epoch 1 using seqex
Epoch: 1 Learning rate: 1.000
0.001 perplexity: 7694.583 speed: 2168 wps
0.200 perplexity: 905.867 speed: 8670 wps
0.400 perplexity: 615.443 speed: 8764 wps
0.600 perplexity: 487.248 speed: 8793 wps
0.800 perplexity: 414.645 speed: 8809 wps
0.999 perplexity: 364.610 speed: 8819 wps
Epoch: 1 Train Perplexity: 364.610
Epoch: 1 Valid Perplexity: 2.718

real	1m55.948s
user	1m14.714s
sys	0m13.987s
======== seqex lglg ============
Starting training from epoch 1 using seqex
Epoch: 1 Learning rate: 1.000
0.001 perplexity: 115883.198 speed: 741 wps
0.200 perplexity: 3240.550 speed: 988 wps
0.400 perplexity: 1805.696 speed: 989 wps
0.600 perplexity: 1266.270 speed: 990 wps
0.800 perplexity: 980.367 speed: 989 wps
0.999 perplexity: 798.548 speed: 990 wps
Epoch: 1 Train Perplexity: 798.548
Epoch: 1 Valid Perplexity: 2.718

real	16m42.652s
user	5m7.490s
sys	11m8.747s

