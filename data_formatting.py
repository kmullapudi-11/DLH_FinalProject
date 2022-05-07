import re
import glob
import os
import fnmatch
from sklearn.model_selection import train_test_split
import pandas as pd

def generate_ast_dictionary_for_file(ast_lines):
	'''
	Dictionary of list of objects keyed by the line number
	{
		start_word,
		end_word,
		assertion
	}
	'''
	ast_dict = {}
	for line in ast_lines:
		# Example line: c="hypertension" 8:2 8:2||t="problem"||a="present"
		# We will extract line = 8, start word = 2, end work = 2, assertion = "present"
		context = line.split("\" ")[0]
		rest = line.split("\" ")[1]
		location_str, context_type, assertion_str = rest.split("||")
		location_str = location_str.strip()
		
		# location str ~ "8:2 8:2"
		start_line, start_word_idx, end_line, end_word_idx = re.split(':| ', location_str)
		start_word_idx = int(start_word_idx)
		end_word_idx = int(end_word_idx)
		start_line = int(start_line)
		end_line = int(end_line)

		# Check values parsed make sense
		if start_line != end_line:
			raise ValueError("start line and end line are different")
		elif start_word_idx > end_word_idx:
			raise ValueError("start word is after end word")

		# assertion_str is in the format a="present"
		assertion = assertion_str.replace("\"", "", 2).replace("a=", "", 1).strip()
		possible_assertions = ["present", "absent", "possible", "conditional", "hypothetical", "associated_with_someone_else"]
		if assertion not in possible_assertions:
			raise ValueError("invalid assertion: " + str(assertion))
		
		if start_line not in ast_dict:
			ast_dict[start_line] = []
		ast_dict[start_line].append({
			"start_word": start_word_idx,
			"end_word": end_word_idx,
			"assertion": assertion
		})
	return ast_dict

"""
Inputs: lines of text and a dictionary of line_num -> assertion/context information
For each of the assertion/context objects, generate a new output sentence that adds
tokens around the context phrase.
Also output a list of labels that match up with the assertions.
"""
def generate_output_lines_and_labels(text_lines, ast_dict):
	# Line numbers are 1 indexed with reference to the ast file.
	line_num = 1

	output_lines = []
	output_labels = []
	for line in text_lines:
		if line_num not in ast_dict:
			line_num += 1
			continue
		line = line.strip()
		for ast in ast_dict[line_num]:
			start_word = ast["start_word"]
			end_word = ast["end_word"]
			assertion = ast["assertion"]
			words_in_line = line.split(" ")
			# Insert the tokens "ENT_1_START""  and "ENT_1_END" around the context words
			words_in_line.insert(end_word + 1, "ENT_1_END")
			words_in_line.insert(start_word, "ENT_1_START")
			output_lines.append(" ".join(words_in_line))
			output_labels.append(assertion)
		line_num += 1

	if len(output_lines) != len(output_labels):
		raise ValueError("Output lines and labels are not of the save size.")
	return output_lines, output_labels

dir_path = os.path.dirname(os.path.realpath(__file__))
print("Current directory: " + dir_path)

# Create a list of pairs for all txt and ast files in the training data.
# The training data is located in two separate folders based on its location.
original_path_ast = "concept_assertion_relation_training_data/beth/ast"
all_records = os.listdir(original_path_ast)
paths = []
for record_ast in all_records:
	record_txt = record_ast.replace(".ast", ".txt")
	path_txt = original_path_ast.replace("/ast", "/txt") + "/" + record_txt
	path_ast = original_path_ast + "/" + record_ast

	paths.append((path_ast, path_txt))

original_path_ast = "concept_assertion_relation_training_data/partners/ast"
all_records = os.listdir(original_path_ast)
for record_ast in all_records:
	record_txt = record_ast.replace(".ast", ".txt")
	path_txt = original_path_ast.replace("/ast", "/txt") + "/" + record_txt
	path_ast = original_path_ast + "/" + record_ast
	paths.append((path_ast, path_txt))

# For each pair of files, generate the lines and labels from those files.
final_output_lines = []
final_output_labels = []
for path_tuple in paths:
	path_ast, path_txt = path_tuple
	text_file = open(path_txt, 'r')
	ast_file = open(path_ast, 'r')
	text_lines = text_file.readlines()
	ast_lines = ast_file.readlines()

	ast_dict = generate_ast_dictionary_for_file(ast_lines)
	output_lines, output_labels = generate_output_lines_and_labels(text_lines, ast_dict)
	final_output_labels.extend(output_labels)
	final_output_lines.extend(output_lines)

X_train, X_test, y_train, y_test = train_test_split(final_output_lines, final_output_labels, test_size=0.30, random_state=42)

train_df = pd.DataFrame()
train_df['tgt'] = y_train
train_df['input'] = X_train
train_df['show_inp'] = X_train
train_df.to_csv('tmp2/train.csv', index=False)

test_df = pd.DataFrame()
test_df['tgt'] = y_test
test_df['input'] = X_test
test_df['show_inp'] = X_test
test_df.to_csv('tmp2/test.csv', index=False)
test_df.to_csv('tmp2/valid.csv', index=False)

"""
Write output to csv file with columns:
label, sentence, sentence
"""
# with open('tmp2/pre_processed_lines.csv', 'w') as f:
#     for idx, item in enumerate(final_output_lines):
#     	item = item.replace("\"", "", 10)
#         f.write("%s,\"%s\",\"%s\"\n" % (final_output_labels[idx], item, item))
