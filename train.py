
def _sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def compute_loss(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = _sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss

def pad_seq(arr, length, pad_token):

    if len(arr) == length:
        return np.array(arr)
    
    return np.concatenate((arr, [pad_token]*(length - len(arr))))
                 

def create_training(start,end,set_batch):
    

    sentence_lens = [len(preprocess_numberize(sentence)) for sentence in set_batch] 

    sorted_indices = sorted(list(range(len(sentence_lens))), key=lambda i: sentence_lens[i], reverse=True)
    set_batch = [set_batch[i] for i in sorted_indices if sentence_lens[i] > 0]
    

    train_id_batch = [train_id for train_id in train_ids[start:end]]
    train_id_batch = [train_id_batch[i] for i in sorted_indices if sentence_lens[i] > 0]
    training_input = [load_image(train_id_to_file[train_id]).squeeze() for train_id in train_id_batch]
    training_input = torch.stack(training_input)
    
    sentence_lens = [sentence_lens[i] for i in sorted_indices if sentence_lens[i] > 0]   
    max_len = max(sentence_lens)                            
                         
    # Preprocess all of the sentences in each batch
    w2v_embedded_list = [preprocess_word2vec(sentence) for sentence in set_batch]
    w2v_embedded_list_padded = [pad_seq(embed, max_len, np.zeros(wordEncodingSize)) 
                                        for embed in w2v_embedded_list]
    numberized_list = [preprocess_numberize(sentence) for sentence in set_batch]
    numberized_list_padded = [pad_seq(numb, max_len, 0).astype(torch.LongTensor) for numb in numberized_list]
    
    one_hot_embedded_list = [preprocess_one_hot(sentence) for sentence in set_batch]
    one_hot_embedded_list_padded = [pad_seq(embed, max_len, np.zeros(vocabularySize)) 
                                        for embed in one_hot_embedded_list]
    
    one_hot_output = Variable(torch.FloatTensor(one_hot_embedded_list_padded)).cuda()
    one_hot_output = one_hot_output.transpose(0, 1)

                
    w2v_input = Variable(torch.FloatTensor(w2v_embedded_list_padded)).cuda()    
    training_output = Variable(torch.LongTensor(numberized_list_padded)).cuda()    
    training_output = training_output.transpose(0, 1)
    w2v_input = w2v_input.transpose(0, 1)

    return training_input,training_output,w2v_input,one_hot_output,sentence_lens



def train(input_variable, 
          target_variable, 
          w2v_input, 
          one_hot_output,
          encoder, 
          decoder, 
          decoder_optimizer, 
          converter_optimizer,
          input_lens,
          criterion, 
          embeddings=w2v_embeddings):
    
    decoder_optimizer.zero_grad()
    converter_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    #print(target_length)

    # Pass through the encoder
    features_output = encoder.features(input_variable)
    classifier_input = features_output.view(batch_size, -1)
    encoder_output = modified_classifier(classifier_input)
    encoder_output = converter(encoder_output).unsqueeze(0)
    
    #print(encoder_output)  
    
    
    # Construct the decoder input (initially <SOS> for every batch)
    decoder_input = Variable(torch.FloatTensor([[embeddings[word2index["<SOS>"]]
                                                for i in range(w2v_input.size(1))]])).cuda()
    #print(decoder_input)

    #print(encoder_output)
    decoder_hidden = encoder_output
    decoder_state = encoder_output

    # Prepare the results tensor
    all_decoder_outputs = Variable(torch.zeros(*one_hot_output.size())).cuda()
    all_decoder_outputs[0] = Variable(torch.FloatTensor([[one_hot_embeddings[word2index["<SOS>"]]
                                                for i in range(w2v_input.size(1))]])).cuda()
        
    # Iterate over the indices after the first.
    for t in range(1,target_length):
        decoder_output,decoder_hidden,decoder_state = decoder(decoder_input,decoder_hidden,decoder_state)
    
        
        if random.random() <= 0.9:
            decoder_input = w2v_input[t].unsqueeze(0)
        else:
            topv, topi = decoder_output.data.topk(1)
                       
            #Prepare the inputs
            decoder_input = torch.stack([Variable(torch.FloatTensor(embeddings[ni])).cuda()
                                         for ni in topi.squeeze()]).unsqueeze(0)
        
        #print(decoder_input)
        #print(decoder_hidden)
        #print(decoder_state)
        # Save the decoder output
        all_decoder_outputs[t] = decoder_output
    
        #print(all_decoder_outputs.transpose(0,1).contiguous())
    
    loss = compute_loss(all_decoder_outputs.transpose(0,1).contiguous(),
                        target_variable.transpose(0,1).contiguous(),
                    Variable(torch.LongTensor(input_lens)).cuda())
    
    loss.backward()    
    torch.nn.utils.clip_grad_norm(decoder.parameters(), 10.0) 
    decoder_optimizer.step()
    converter_optimizer.step()

    return loss.data[0]


