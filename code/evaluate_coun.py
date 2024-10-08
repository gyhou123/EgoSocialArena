import random
import tqdm
import argparse
import prompts_tomi
from sim_utils_tomi import *
import collections
import json
from collections import Counter

random.seed(0)


def most_common(lst):
    data = collections.Counter(lst)
    return data.most_common(1)[0][0]

def read_txt(filename):
    with open(filename, 'r') as f:
        s = f.read()
    return s

def evaluate_coun():
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

        table = wandb.Table(columns=["story", "question", "question_type", "label", "guess","correct"])
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
    
    # Print stuff
    print("\n------------------------")
    print("    EVALUATING SOCIAL    ")
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
        elif "llama" in args.eval_model:
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
            #story, question, label, containers, story_type, question_type = fields["story"], fields["question"], fields["answer"], fields["containers"], fields["story_type"], fields["question_type"]
            story, question, label, containers_list= fields["Story"], fields["Question"], fields["Answer"], fields["Option"]
            question_type = "coun"
            if args.category != "all" and question_type not in args.category.split(","):
                continue
            
            i += 1
            if i > args.num_probs:
                break
            
        
            baselinePrompt = prompts_tomi.baselineparPrompt.format(story=story, question=question, containers_list=containers_list)

            # If category is specified, then only evaluate that question type.
            
            
            #baselinePrompt = prompts_tomi.baselinePrompt.format(story=story, question=question, containers_0=containers[0], containers_1=containers[1])
            
            
            
            #if "llama" in args.sim_model:
                # We have separate prompts for these because, again, llama is stubborn.
                # Nothing too crazy, we just start off its response for it.
                #baselinePrompt = prompts_tomi.llamaprompt.format(story=story, question=question, containers_0=containers[0], containers_1=containers[1])
                #cotPrompt = prompts_tomi.llamaCotPrompt.format(story=story, question=question, containers_0=containers[0], containers_1=containers[1])
            
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
            
            '''
            # Grade answer
            correct = None
            if label in prediction.lower():
                correct = True
            else:
                correct = False
            
            if correct:
                correctNum += 1
            totalNum += 1
            '''

            #Evaluation 
            EvaluateModel = ChatGPT_Evaluate('gpt-4o',temperature=args.temperature, verbose=args.verbose)

            Evaluate_prompt = f"""\
This is someone's response to a question:

{prediction}

This is the correct answer:

{label}

Is their final answer correct? Output 'True' or 'False' only. If they chose option c) or said something like "neither", output 'False'. If they consider both options but ultimately don't decide, also output 'False'.
"""
            Evaluate_Result = EvaluateModel.getOutput(Evaluate_prompt)
            correct = None
            if Evaluate_Result == 'True':
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
                table.add_data(story, question, question_type, label, prediction, correct)

            
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
    #parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--eval_model', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct-Turbo')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--num_probs', '-n', type=int, default=50)
    parser.add_argument('--max_tokens', type=int, default=300)
    parser.add_argument('--method', type=str, default='baseline')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--eight_bit', action='store_true')
    parser.add_argument('--wandb', type=int, default=0)
    parser.add_argument('--tags', type=str, default="social4baseline")
    parser.add_argument('--category', type=str, default='all')
    parser.add_argument('--perspective_model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--sim_model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--entity', type=str, default=None)
    global args
    args = parser.parse_args()
    
    # Evaluate on PaR
    evaluate_coun()

if __name__ == '__main__':
    main()