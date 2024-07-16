# Bilevel Black-Box Prompt Optimization for LLM (BLBPO)

## 🚀 Get started

### Check Data

```javascript
|-- BLBPO
	|-- data
    	|-- 5G
			|-- train.pt
			|-- test.pt
			|-- example.pt
			|-- vocabulary.pt
    	|-- MRPC
			...
    	|-- SST
			...
    |-- utils
		|-- llm.py
		|-- log.py
    |-- BLBPO-5G.py
    |-- BLBPO-MRPC.py
    |-- BLBPO-SST.py
    |-- README.md
    |-- requirements.txt
    ......
```

### Install 

```python
# Please install requirements.txt in a Python>=3.8.0 environment, including PyTorch>=1.8. 
pip install -r requirements.txt  # install
```

## 🏃Training

```python
# Dataset-5G
python BLBPO-5G.py --just_test False
python BPT-5G.py --just_test False
python ICLT-5G.py --just_test False
# Dataset-MRPC
python BLBPO-MRPC.py --just_test False
......
```

## ☘️Testing

```python
# Dataset-5G
python BLBPO-5G.py --just_test True
python BPT-5G.py --just_test True
python ICLT-5G.py --just_test True
# Dataset-MRPC
python BLBPO-MRPC.py --just_test True
......
```

