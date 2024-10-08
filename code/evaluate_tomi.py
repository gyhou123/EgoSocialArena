

import random
import tqdm
import argparse
import prompts_tomi
from sim_utils_tomi import *
import collections
import json
from collections import Counter


DATA_DIR = '../data/fixedtomi/test_balanced.jsonl'
random.seed(0)


def most_common(lst):
    data = collections.Counter(lst)
    return data.most_common(1)[0][0]

def read_txt(filename):
    with open(filename, 'r') as f:
        s = f.read()
    return s

def evaluate_tomi():
    # Load the dataset
    gold = []             # Ground truths
    predictions = []      # Predictions
    results = []          # Results (accuracy)
    category_results = {} # Results per category
    category_percents = {}
    bad = 0
    correctNum = 0
    totalNum = 0
    false_case = []
    false_question_type = []
    index_false = []

    if args.wandb == 1:
        run = wandb.init(
            project=args.project,
            entity=args.entity, 
            config=vars(args),
            tags=args.tags.split(','),
        )

        table = wandb.Table(columns=["story", "perspective", "question", "story_type", "question_type", "label", "guess","correct"])
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
    
    # Print stuff
    print("\n------------------------")
    print("    EVALUATING TOMI      ")
    print("------------------------")
    if args.eval_model == None:
        print(f"PER MODEL: {args.perspective_model}")
        print(f"SIM MODEL: {args.sim_model}")
    else:
        print(f"EVAL MODEL: {args.eval_model}")
    print(f"DATA: {args.data_dir}")
    print(f"METHOD: {args.method}")
    print(f"CATEGORY: {args.category}")
    print(f"N = {args.num_probs}")
    print("------------------------\n")

    
    if args.eval_model == None:
        if "gpt" in args.perspective_model:
            perspectiveModel = ChatGPT(args.perspective_model, temperature=args.temperature, verbose=args.verbose)
        else:
            perspectiveModel = LLM(args.perspective_model,load_in_8bit=args.eight_bit, temperature=args.temperature, verbose=args.verbose)
        
        if "gpt" in args.sim_model:
            simModel = ChatGPT(args.sim_model, temperature=args.temperature, verbose=args.verbose)
        else:
            simModel = LLM(args.sim_model,load_in_8bit=args.eight_bit, temperature=args.temperature, verbose=args.verbose)

    else:
        simModel = None
        if "gpt" in args.eval_model or "o1" in args.eval_model:
            perspectiveModel = ChatGPT(args.eval_model, temperature=args.temperature, verbose=args.verbose)
        elif "claude" in args.eval_model:
            perspectiveModel = Claude(args.eval_model,temperature=args.temperature, verbose=args.verbose)
        elif "llama" in args.eval_model or "mistral" in args.eval_model:
            perspectiveModel = LLaMA(args.eval_model,temperature=args.temperature, verbose=args.verbose)
        else:
            perspectiveModel = LLM(args.eval_model,load_in_8bit=args.eight_bit, temperature=args.temperature, verbose=args.verbose)

    
    # Query model on dataset
    with open(args.data_dir) as f_in:
        i = 0
        for index, line in tqdm.tqdm(enumerate(f_in), total=min(len(read_txt(args.data_dir).split('\n')), args.num_probs)):
            # Parse each line as JSON
            fields = json.loads(line.strip())
            
            # extract fields from JSON
            story, question, label, containers, story_type, question_type = fields["story"], fields["question"], fields["answer"], fields["containers"], fields["story_type"], fields["question_type"]
            if story_type != "true_belief" and (question_type == "first_order_0_tom" or question_type == "first_order_1_tom" or question_type == "first_order_0_no_tom" or question_type == "first_order_1_no_tom"):
                continue

            if args.category != "all" and question_type not in args.category.split(","):
                continue
            
            i += 1
            if i > args.num_probs:
                break

            story = story.split('\n')
            for k in range(len(story)):
                words = story[k].split()
                del words[0]
                story[k] = " ".join(words)
            
            def merge_enter(story):
                x = 0
                while x < len(story)-2:
                    words1 = story[x].split()
                    words2 = story[x+1].split()
                    words3 = story[x+2].split()

                    if words1[1] == words2[1] == words3[1] and words1[-1] == words2[-1] == words3[-1]:
                        merged_enter = ' '.join(words1 + words2 + words3)
                        story[x] = merged_enter
                        del story[x+1]
                        del story[x+1]
                        
                    elif words1[1] == words2[1] and words1[-1] == words2[-1]:
                        merged_enter = ' '.join(words1 + words2)
                        story[x] = merged_enter
                        del story[x+1]
                        
                    elif words1[1] == words3[1] and words1[-1] == words3[-1]:
                        merged_enter = ' '.join(words1 + words3)
                        story[x] = merged_enter
                        del story[x+2]
                        
                    elif words2[1] == words3[1] and words2[-1] == words3[-1]:
                        merged_enter = ' '.join(words2 + words3)
                        story[x+1] = merged_enter
                        del story[x+2]
                        
                    else:
                        x+=1
                return story


            def merge_story(story):
                y = 0
                while y < len(story)-1:
                    words1 = story[y].split()
                    words2 = story[y+1].split()
                    
                    words1[-1] = words1[-1].replace(".","")
                    if words1[-1] == words2[1]:
                        words1[-1] = words1[-1] + ","
                        merged_story = ' '.join(words1 + words2)
                        story[y] = merged_story
                        del story[y+1]
                        y+=1
                    else:
                        y+=1
                return story

            story = merge_enter(story)
            story = merge_story(story)
            fields["story"] = story

            '''
            def extract_last_words(sentences):
                last_words = []
                for sentence in sentences:
                # Split the sentence and take the last word after splitting by spaces and removing punctuation
                    last_word = sentence.split()[-1].strip('.')
                    last_words.append(last_word)
                return last_words
            
            last_words = extract_last_words(story)
            most_common_element = Counter(last_words).most_common(1)[0][0]

            #获取初始需要转换视角的角色
            if question_type == "first_order_0_tom" or question_type == "first_order_1_tom" or question_type == "first_order_0_no_tom" or question_type == "first_order_1_no_tom":
                character = question.split(" ")[2]
            else:
                character = question.split(" ")[2]
                temp_char = character + ' exited the ' + most_common_element + '.'
                #temp_char在story的中间元素而不是全部story范围内
                #print(story[1:-1])
                if temp_char in story[1:-1]:
                    character = question.split(" ")[5]
            #story第三视角转换成第一视角
            #将story中和character相同的字符串替换为you
            for q in range(len(story)):
                words = story[q].split()
                for n in range(len(words)):
                    if words[n] == character:
                        words[n] = "you"
                        story[q] = " ".join(words)

            #question第三视角转换成第一视角
            #将question中和character相同的字符串替换为you
            words = question.split()
            for z in range(len(words)):
                if words[z] == character:
                    words[z] = "you"
                    question = " ".join(words)
            
            '''


            # If category is specified, then only evaluate that question type.
            
            
            baselinePrompt = prompts_tomi.baselinePrompt.format(story=story, question=question, containers_0=containers[0], containers_1=containers[1])
            rulesPrompt = prompts_tomi.rulesPrompt.format(story=story, question=question, containers_0=containers[0], containers_1=containers[1])
            oneShotPrompt = prompts_tomi.oneShotPrompt.format(story=story, question=question, containers=containers)
            cotPrompt = prompts_tomi.cotPrompt.format(story=story, question=question, containers_0=containers[0], containers_1=containers[1])
            rulesCoTPrompt = prompts_tomi.rulesCoTPrompt.format(story=story, question=question, containers_0=containers[0], containers_1=containers[1])
            oneShotCotPrompt = prompts_tomi.oneShotCotPrompt.format(story=story, question=question, containers=containers)
            questionPrompt = prompts_tomi.questionPrompt.format(story=story, question=question, containers_0=containers[0], containers_1=containers[1])
            
            
            if "llama" in args.sim_model:
                # We have separate prompts for these because, again, llama is stubborn.
                # Nothing too crazy, we just start off its response for it.
                baselinePrompt = prompts_tomi.llamaprompt.format(story=story, question=question, containers_0=containers[0], containers_1=containers[1])
                cotPrompt = prompts_tomi.llamaCotPrompt.format(story=story, question=question, containers_0=containers[0], containers_1=containers[1])
            
            if args.method == "baseline":
                perspective = "N/A (Baseline)"
                prediction = perspectiveModel.getOutput(baselinePrompt)
                
            elif args.method == "baselineRules":
                perspective = "N/A (Baseline)"
                prediction = perspectiveModel.getOutput(rulesPrompt)
                
            elif args.method == "oneshot":
                perspective = "N/A (One Shot)"
                prediction = perspectiveModel.getOutput(oneShotPrompt)
                
            elif args.method == "cot":
                perspective = perspectiveModel.getOutput(cotPrompt)
                try:
                    prediction = perspective.split("Answer:")[1]
                except:
                    prediction = perspective[-30:]
                perspective = "Chain of Thought: \n" + perspective
                
            elif args.method == "cotRules":
                perspective = perspectiveModel.getOutput(rulesCoTPrompt)
                try:
                    prediction = perspective.split("Answer:")[1]
                except:
                    prediction = perspective[-30:]
                perspective = "Chain of Thought: \n" + perspective
                
            elif args.method == "oneshotcot":
                perspective = perspectiveModel.getOutput(oneShotCotPrompt)
                prediction = perspective.strip().split(".")[-2].strip()
                
            elif args.method == "onePromptSimulation":
                prediction, perspective = oneBigPrompt(perspectiveModel, story, questionPrompt)
                
            elif args.method == "simulation":
                prediction, perspective = evalQuestion(perspectiveModel, fields, questionPrompt, simModel=simModel)
                
            else:
                print("Please specify a valid evaluation method!")
                return
            
            
            # Grade answer
            correct = None
            if label in prediction.lower():
                correct = True
            else:
                correct = False
            
            if correct:
                correctNum += 1
            totalNum += 1
            
            

            if args.verbose:
                print(f"\n### Correct: {correct} ###\n")

            # if not correct:
            #     # This means the model got it wrong.
            if args.wandb == 1:
                table.add_data(story, perspective, question, story_type, question_type, label, prediction, correct)

            
            # Append ground truth and model prediction.
            gold.append(label)
            predictions.append(prediction)
            
            # Calculate category result
            temp = category_results.get("count"+question_type, {"correct": 0, "total" : 0})
            if correct:
                temp["correct"] += 1
            temp["total"] += 1
            percent = temp["correct"] / temp["total"]
            category_results["count"+question_type] = temp
            category_percents[question_type] = percent
            
            if prediction == -1:
                bad += 1


            
            if correct == False:
                false_case.append(story)
                false_question_type.append(question_type)
                index_false.append(index)
                    
            
            if args.wandb == 1:
                try:
                    rollingAccuracy = correctNum / totalNum
                    rollingAccuracy_dict = {'running_acc' : rollingAccuracy}
                    wandb.log({**category_percents, **rollingAccuracy_dict, **category_results})
                except:
                    pass


    accuracy = correctNum / totalNum
    print(correctNum)
    print(totalNum)
    print(f"Accuracy: {accuracy*100:.3f}%")
    results.append(f"Accuracy : {accuracy:.3f}")
    print(f"Bad responses: {bad}")
    results.append(f"Bad responses: {bad}")

    #with open('story.json', 'w') as json_file:
        #json.dump(false_case, json_file)
    
    #with open('question_type.json', 'w') as json_file:
        #json.dump(false_question_type, json_file)
    
    #with open('index.json', 'w') as json_file:
        #json.dump(index_false, json_file)


    if args.wandb == 1:
        wandb.log({'total_acc' : accuracy})
        wandb.log({'bad_responses' : bad})
        wandb.log({'wrong_answers': table})
        wandb.finish()




    # Print results
    print("\n------------------------")
    print("         RESULTS        ")
    print("------------------------")
    print(f"MODEL: {args.eval_model}")
    print(f"METHOD: {args.method}")
    print(f"ACCURACY: {accuracy:.2%}")
    print("------------------------\n")
    print(gold)
    print(predictions)
    



def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='')
    #parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--eval_model', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--num_probs', '-n', type=int, default=300)
    parser.add_argument('--max_tokens', type=int, default=300)
    parser.add_argument('--method', type=str, default='baseline')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--eight_bit', action='store_true')
    parser.add_argument('--wandb', type=int, default=0)
    parser.add_argument('--tags', type=str, default="debug")
    parser.add_argument('--category', type=str, default='all')
    parser.add_argument('--perspective_model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--sim_model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--entity', type=str, default=None)
    global args
    args = parser.parse_args()
    
    # Evaluate on ToMi
    evaluate_tomi()

if __name__ == '__main__':
    main()