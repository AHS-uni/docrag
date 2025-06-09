#!/usr/bin/env bash

ROOT=data                     # base folder holding each *-sample directory
USER=AHS-uni                 # HF user/org
MSG="Uploaded samples."   # commit message for all pushes
PRIV=""                       # "--private" if you want private repos

##############################################################################
# 1) ArxivQA
##############################################################################
python scripts/run/push_to_hub.py \
       -r "$ROOT/arxivqa-sample"  -n "$USER/arxivqa-corpus-sample" -t corpus -m "$MSG" $PRIV

python scripts/run/push_to_hub.py \
       -r "$ROOT/arxivqa-sample"  -n "$USER/arxivqa-qa-sample"     -t qa \
       -s train=unified_qas/arxivqa.jsonl \
       -m "$MSG" $PRIV

##############################################################################
# 2) DUDE
##############################################################################
python scripts/run/push_to_hub.py \
       -r "$ROOT/dude-sample"  -n "$USER/dude-corpus-sample" -t corpus -m "$MSG" $PRIV

python scripts/run/push_to_hub.py \
       -r "$ROOT/dude-sample"  -n "$USER/dude-qa-sample"     -t qa \
       -s train=unified_qas/2023-03-23_DUDE_gt_test_PUBLIC_train.jsonl \
       -s test=unified_qas/2023-03-23_DUDE_gt_test_PUBLIC_test.jsonl \
       -m "$MSG" $PRIV


##############################################################################
# 4) MMLongBench-Doc
##############################################################################
python scripts/run/push_to_hub.py \
       -r "$ROOT/mmlongbenchdoc-sample"  -n "$USER/mmlongbenchdoc-corpus-sample" -t corpus -m "$MSG" $PRIV

python scripts/run/push_to_hub.py \
       -r "$ROOT/mmlongbenchdoc-sample"  -n "$USER/mmlongbenchdoc-qa-sample"     -t qa \
       -s train=unified_qas/mmlongbenchdoc.jsonl \
       -m "$MSG" $PRIV

##############################################################################
# 5) MP-DocVQA
##############################################################################
python scripts/run/push_to_hub.py \
       -r "$ROOT/mpdocvqa-sample"  -n "$USER/mpdocvqa-corpus-sample" -t corpus -m "$MSG" $PRIV

python scripts/run/push_to_hub.py \
       -r "$ROOT/mpdocvqa-sample"  -n "$USER/mpdocvqa-qa-sample"     -t qa \
       -s train=unified_qas/train.jsonl \
       -s val=unified_qas/val.jsonl \
       -s test=unified_qas/test.jsonl \
       -m "$MSG" $PRIV

##############################################################################
# 7) TATDQA
##############################################################################
python scripts/run/push_to_hub.py \
       -r "$ROOT/tatdqa-sample"  -n "$USER/tatdqa-corpus-sample" -t corpus -m "$MSG" $PRIV

python scripts/run/push_to_hub.py \
       -r "$ROOT/tatdqa-sample"  -n "$USER/tatdqa-qa-sample"     -t qa \
       -s train=unified_qas/train.jsonl \
       -s dev=unified_qas/dev.jsonl \
       -s test=unified_qas/test.jsonl \
       -m "$MSG" $PRIV
