open_project aes_trial_2

add_files aes.c
set_top aes

open_solution solution
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 10 -name default

set_directive_array_partition -type cyclic -factor 64 -dim 1 "aes_trial_2" sbox
set_directive_array_partition -type cyclic -factor 8 -dim 1 "aes_trial_2" k
set_directive_array_partition -type cyclic -factor 4 -dim 1 "aes_trial_2" buf

set_directive_pipeline "aes_trial_2/sub"
set_directive_unroll -factor 8 "aes_trial_2/sub"
set_directive_pipeline "aes_trial_2/addkey"
set_directive_unroll -factor 8 "aes_trial_2/addkey"
set_directive_pipeline "aes_trial_2/cpkey"
set_directive_unroll -factor 8 "aes_trial_2/cpkey"
set_directive_pipeline "aes_trial_2/mix"
set_directive_unroll -factor 2 "aes_trial_2/mix"
set_directive_pipeline -off "aes_trial_2/exp1"

set_directive_pipeline -off "aes_trial_2/exp2"

set_directive_pipeline "aes_trial_2/ecb1"
set_directive_unroll -factor 8 "aes_trial_2/ecb1"
set_directive_pipeline -off "aes_trial_2/ecb2"

set_directive_pipeline -off "aes_trial_2/ecb3"


csynth_design
exit
