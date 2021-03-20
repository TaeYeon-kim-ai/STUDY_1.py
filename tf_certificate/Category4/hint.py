with open ('sarcasm.json', 'r') as f :
    dataset = json.load(f)

    for item in datasets :
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])


x_train = sentences[0:training_size]
x_test = sentences[training_size:]
y_train = labels[0:training_size]
y_test = labels[training_size:]
