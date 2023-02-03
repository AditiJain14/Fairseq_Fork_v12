## Fairseq_Fork_v12

An off-the-shelf working version of Fairseq's monotonic multihead attention for SiMT

### To get this running
Clone this repo first
* `git clone https://github.com/AditiJain14/Fairseq_Fork_v12.git`
* `cd monotonic_multihead_attention`

Install fairseq inside this repo
* `git clone -b monotonic_multihead_attention https://github.com/facebookresearch/fairseq.git`
* `cd fairseq`
* `pip3 install -e .`

Then, `cd run_scripts` from this directory.

Set the path to this repo in the`ROOT` variable in each of the bash scripts.

Finally, run the following scripts:
* `bash prepare-iwslt14.sh`
* `bash train.sh`
