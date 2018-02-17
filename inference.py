#val_images and ground truth captions 
decoder.load_state_dict(torch.load('./decoder4new.pth'))
converter.load_state_dict(torch.load('./converter4new.pth'))
encoder.load_state_dict(torch.load('./encoder4new.pth'))

#encoder.eval()

# Your code goes here
def map_inference(input_variable, embeddings=w2v_embeddings, max_length=20):
    
    features_output = encoder.features(input_variable)
    classifier_input = features_output.view(1, -1)
    encoder_output = modified_classifier(classifier_input)
    encoder_output = converter(encoder_output).unsqueeze(0)

    #print(encoder_output)

    # Construct the decoder input (initially <SOS> for every batch)
    decoder_input = Variable(torch.FloatTensor([[embeddings[word2index["<SOS>"]]]])).cuda()
    #print(decoder_input)
    decoder_hidden = encoder_output
    decoder_state = encoder_output
    
    # Iterate over the indices after the first.
    decoder_outputs = []
    for t in range(1,max_length):
        decoder_output,decoder_hidden,decoder_state = decoder(decoder_input,decoder_hidden,decoder_state)
    
        # Get the top result
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        decoder_outputs.append(ni)

        if vocabulary[ni] == "<EOS>":
            break
        
        #Prepare the inputs
        decoder_input = Variable(torch.FloatTensor([[embeddings[ni]]])).cuda()

    return ' '.join(vocabulary[i] for i in decoder_outputs)

#send val images 
for val_id in val_ids[:5]:    
    img_input = load_image(val_id_to_file[val_id])
    caption = map_inference(img_input)
    print(caption)
    print(val_id_to_captions[val_id][0])
    print(" ")



# Your code goes here
def sample_inference(input_variable, embeddings=w2v_embeddings, max_length=20):
    
    features_output = encoder.features(input_variable)
    classifier_input = features_output.view(1, -1)
    encoder_output = modified_classifier(classifier_input)
    encoder_output = converter(encoder_output).unsqueeze(0)
    

    # Construct the decoder input (initially <SOS> for every batch)
    decoder_input = Variable(torch.FloatTensor([[embeddings[word2index["<SOS>"]]]])).cuda()
    decoder_hidden = encoder_output
    decoder_state = encoder_output
    
    # Iterate over the indices after the first.
    decoder_outputs = []
    for t in range(1,max_length):
        decoder_output,decoder_hidden,decoder_state = decoder(decoder_input,decoder_hidden,decoder_state)
        probs = np.exp(decoder_output.data[0].cpu().numpy())
        sample_sum = probs[0]
        random_sample = random.random()
        ni = 0
        while sample_sum < random_sample:
            ni += 1
            sample_sum += probs[ni]
            
        decoder_outputs.append(ni)

        if vocabulary[ni] == "<EOS>":
            break
        
        #Prepare the inputs
        decoder_input = Variable(torch.FloatTensor([[embeddings[ni]]])).cuda()

    return ' '.join(vocabulary[i] for i in decoder_outputs)

#send val images #send val images 
for val_id in val_ids[:5]:    
    img_input = load_image(val_id_to_file[val_id])
    caption = sample_inference(img_input)
    print(caption)
    print(val_id_to_captions[val_id][0])
    print(" ")

        