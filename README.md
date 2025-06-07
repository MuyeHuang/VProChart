# VProChart: Answering Chart Question through Visual Perception Alignment Agent and Programmatic Solution Reasoning

[![ModelScope](https://img.shields.io/badge/ModelScope-VProChart-blue)](https://www.modelscope.cn/models/HUANGMUYE/VProChart-VPAgent)  [![arXiv](https://img.shields.io/badge/arXiv-2409.01667-red)](https://arxiv.org/abs/2409.01667)    [![AAAI-25](https://img.shields.io/badge/AAAI--25-Accepted-brightgreen)](https://ojs.aaai.org/index.php/AAAI/issue/view/627)


## 🎉 News

- **2024.12.09**: Our paper has been accepted by AAAI-25.



## 🚀 Quick Start

### 📥 Model Download

Click the badge below to grab the pretrained VPAgent model from ModelScope:

[![Download on ModelScope](https://img.shields.io/badge/Download-VProChart_VPAgent-blue.svg)](https://www.modelscope.cn/models/HUANGMUYE/VProChart-VPAgent)

---

### 🛠️ Dependencies

Install the exact versions for full compatibility:

```bash
pip install transformers==4.28.1 \
            pytorch-lightning==1.8.5 \
            datasets \
            sentencepiece
```

---

### 🎯 Usage (Inference)

For a quick example of loading and querying VProChart, see our test script:

[![View test_vpagent.py](https://img.shields.io/badge/View-test__vpagent.py-green)](https://github.com/MuyeHuang/VProChart/blob/main/test_vpagent.py)

Programmatic Solution Reasoning is still being organized and will be released soon.


### 🔧 Finetuning

To adapt VProChart on your own data, consult the finetuning script:

[![View finetune_chartqa.py](https://img.shields.io/badge/View-finetune__chartqa.py-orange)](https://github.com/MuyeHuang/VProChart/blob/main/finetune_chartqa.py)

**Example CLI**  
```bash
python finetune_chartqa.py \
  --data-path "your_hf_dataset_name_or_local_path" \
  --train-images "/path/to/train/images/" \
  --valid-images "/path/to/val/images/" \
  --output-dir "./finetuned_model_output/" \
  --max-steps 40000 \
  --batch-size 8 \
  --valid-batch-size 1 \
  --max-length 512 \
  --num-workers 12 \
  --lr 5e-5 \
  --check-val-every-n-epoch 1 \
  --log-every-n-steps 50 \
  --warmup-steps 100 \
  --checkpoint-steps 7000 \
  --gradient-clip-val 1.0 \
  --accumulate-grad-batches 1 \
  --gpus-num 1 \
  --nodes-num 1 \
  --checkpoint-path "/path/to/vprochart_pretrained/"
```
## 📖 Citation

If you find VProChart useful in your research, please cite:

```bibtex
@misc{huang2024vprochartansweringchartquestion,
  title = {VProChart: Answering Chart Question through Visual Perception Alignment Agent and Programmatic Solution Reasoning},
  author = {Muye Huang and Lingling Zhang and Lai Han and Wenjun Wu and Xinyu Zhang and Jun Liu},
  year = {2024},
  eprint = {2409.01667},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2409.01667},
}
```

---

## 🔗 Resources

- **Paper**: [VProChart: Answering Chart Question through Visual Perception Alignment Agent and Programmatic Solution Reasoning](https://arxiv.org/abs/2409.01667)  
- **ModelScope Hub**: [VProChart-VPAgent](https://www.modelscope.cn/models/HUANGMUYE/VProChart-VPAgent)

---

## 🙏 Acknowledgements

This work is partially based on the UniChart project: [vis-nlp/UniChart](https://github.com/vis-nlp/unichart)
