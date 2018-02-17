# Your code goes here
total_bleu = 0
for val_id in val_ids:
    #load image
    img_input = load_image(val_id_to_file[val_id])
    predicted = "<SOS>" + map_inference(img_input)
    
    #load all captions 
    cap_set = [[] for x in range(5)]
    for n in range(5):
        cap_set[n].append(val_id_to_captions[val_id][n])
    
    temp_bleu = 0
    for n in range(5):
        sentence = cap_set[n][0]
        bleu_score = compute_bleu(sentence, predicted)
        temp_bleu += bleu_score
    
    total_bleu += temp_bleu/5
    
print(total_bleu/len(val_ids))
