import timm

models = timm.list_models()
# for model in models:
#     print(model)
backbone = timm.create_model('hrnet_w18', pretrained=True)
print(backbone)