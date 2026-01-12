from one_eval.toolkits.hf_dataset_structure_tool import HFDatasetStructureTool

tool = HFDatasetStructureTool()
info = tool.probe("openai/gsm8k", include_features=True, include_num_examples=True)

# 把 info 直接塞进 state, 再统一交给 agent 编排
print(info)
