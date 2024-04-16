# Challenge 1 - Create an Azure Machine Learning job

```bash
az login
az ml job create --file job.yml --web --resource-group {{rg-name}} --workspace-name {{ws-name}}
```