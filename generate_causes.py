import openai
from tools import *
from cfg import *
from datasets import *
from tqdm import tqdm
import time

# Initialization
dataset_name = 'oxford_flowers' # Change the Name of Task
task_name = {
	'caltech101': 'object',
	'dtd': 'texture',
	'eurosat': 'remote sensing land cover',
	'fgvc': 'aircraft model',
	'food101': 'food',
	'oxford_flowers': 'flower',
	'oxford_pets': 'pet',
	'resisc45': 'remote sensing scene',
	'stanford_cars': 'fine-grained automobile',
	'sun397': 'scene',
	'ucf101': 'action',
}
cause_name = {
	'caltech101': ['shape', 'color', 'texture'],
	'dtd': ['pattern', 'coarseness/granularity', 'contrast', 'directionality', 'regularity', 'roughness', 'entropy'],
	'eurosat': ['texture', 'spectral information', 'spatial characteristics'],
	'fgvc': ['shape and structure','size and proportions', 'surface features', 'engine configuration', 'wing configuration', 'tail design', 'landing gear configuration'],
	'food101': ['texture', 'shape and composition', 'color'],
	'oxford_flowers': ['flower structure', 'shape of the petals', 'color',  'size of the Flower', 'fragrance', 'leaf characteristics', 'flowering time and growth habit'],
	'oxford_pets': ['fur or coat type', 'species and body shape', 'facial features'],
	'resisc45': ['spectral information', 'texture', 'spatial characteristics'],
	'stanford_cars': ['front and rear design', 'shape and body style', 'wheels and rims'],
	'sun397': ['spatial layout and composition', 'texture', 'color and lighting'],
	'ucf101': ['motion and poses', 'contextual elements', 'intensity and speed'],
}


openai.api_key = "" # Please Input your API Key Before Running
_, _, category_list, _ = build_loader(dataset_name, DOWNSTREAM_PATH)
all_responses = {}

# Begin Generating
for i in range(0, 3):
	json_name = "causes/" + dataset_name + "_cse_" + str(i) + ".json"
	for category in tqdm(category_list):
		categoryname = category.replace('.', '')
		prompts = []
		prompts.append("Describe the " + cause_name[dataset_name][i-1] + " of the " + task_name[dataset_name] + " image '" + categoryname + "'")
		all_result = []
		for curr_prompt in prompts:
			response = openai.ChatCompletion.create(
				model="gpt-4o-mini",
				messages=[
					{"role": "system", "content": "You are a helpful assistant."},
					{"role": "user", "content": curr_prompt}
				],
				max_tokens=50,
				temperature=.99,
				n=20,
				stop="."
			)
			for r in range(len(response["choices"])):
				result = response["choices"][r].message.content
				all_result.append(result)
		all_responses[category] = all_result

	with open(json_name, 'w') as f:
		json.dump(all_responses, f, indent=4)
