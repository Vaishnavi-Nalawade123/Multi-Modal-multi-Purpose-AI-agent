from .classifiers.ml_based.test import predict_intent
from .classifiers.rule_based.rule_based_intent_classifier import classify_intent as rule_based_classify_intent
from .classifiers.transformer_based.intend_classifier import predict_intent as transformer_predict_intent
from .classifiers.llm_based.llama_based_intent_classifier import classify_intent_using_llama
def classify_intent(text, method='ml'):

    if method == 'ml':
        return predict_intent(text)
    elif method == 'rule_based':
        return rule_based_classify_intent(text)
    elif method == 'transformer':
        return transformer_predict_intent(text)
    elif method == 'llm':
        return classify_intent_using_llama(text)
    else:
        raise ValueError("Invalid method specified. Choose from 'ml', 'rule_based', or 'transformer'.")

if __name__ == "__main__":
    method = 'ml'
    print(f"using {method} method for intent classification")
    while True:
        user_input = input("Enter your query (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        intent = classify_intent(user_input, method=method)
        print(f"Input: {user_input} --> Intent: {intent}")
