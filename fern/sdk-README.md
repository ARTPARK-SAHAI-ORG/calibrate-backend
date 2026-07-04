# Calibrate Python SDK

Python client for the [Calibrate](https://pense-backend.artpark.ai) API.

## Installation

```sh
pip install calibrate-python-sdk
```

## Usage

```python
from calibrate_python_sdk import Calibrate

client = Calibrate(api_key="sk_your_api_key")

agents = client.agents.list()
```

This SDK is generated from Calibrate's public API with [Fern](https://buildwithfern.com).
