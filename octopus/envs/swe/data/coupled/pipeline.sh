#!/bin/bash
# pipeline.sh: baselines (train, then eval) -> builds (train 25 per repo, eval 45; 32 processes nice 19) with judge loops
# running alongside (train 60 workers, eval 40: about 100 gateway calls at once) -> stage2 for eval, then train (containers one stage at a time)
W=/mnt/disk6/user/coupled-s1a
cd $W
unset XAI_API_KEY GROK_MODELS_BASE_URL  # only judge.sh loads the Grok key (~/.grok/xai.env)
T=$(awk '$1=="train"{print $2}' keys.txt | tr '\n' ' '); E=$(awk '$1=="eval"{print $2}' keys.txt | tr '\n' ' ')
scripts/base.sh coupled-b-train $T > logs/base-train.log 2>&1
scripts/base.sh coupled-b-eval $E > logs/base-eval.log 2>&1
setsid scripts/judge_loop.sh coupled-b-train logs/build-train.log 60 > logs/judge-loop-coupled-b-train.log 2>&1 < /dev/null &
setsid scripts/judge_loop.sh coupled-b-eval logs/build-eval.log 40 > logs/judge-loop-coupled-b-eval.log 2>&1 < /dev/null &
scripts/build.sh coupled-b-train 25 $T > logs/build-train.log 2>&1
scripts/build.sh coupled-b-eval 45 $E > logs/build-eval.log 2>&1
scripts/stage2.sh coupled-b-eval > logs/stage2-eval.log 2>&1
scripts/stage2.sh coupled-b-train > logs/stage2-train.log 2>&1
echo "exit pipeline"
