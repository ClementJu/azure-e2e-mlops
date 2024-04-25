# Challenge 1 - Create an Azure Machine Learning job

```bash
az login
az ml job create --file job.yml --web --resource-group {{rg-name}} --workspace-name {{ws-name}}
```

# Create a Service Principal

```bash
az ad sp create-for-rbac --name {{sp-name}} --role contributor --scopes /subscriptions/{{subscription-id}}/resourceGroups/{{rg-name}}/providers/Microsoft.MachineLearningServices/workspaces/{{ws-name}} --sdk-auth
```