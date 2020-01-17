# Noise-label-generation-and-relabeling
generate noisy label for dataset given noise rate and use the relabel algorithm to relabel these noisy labels

f1.test.py generate noisy labels for all_tickets.csv dataset,use 'body' as feature,'urgency' and 'ticket_type' as predict label. We assume the 'urgency' label may have some noisy labels so we add noise to 'urgency'.We split 48000 data for train dataset and the left for test data.After generate noisy label 's' for 'urgency',we integrate 'body','ticket_type' and 's' to train_data.csv.In classification.py we produce the relabeled label 'relabel'and write it into the relabel.csv file.In addition to the tickets dataset,we also test other datasets.

|dataset|file|result folder|
|:--------:|:---------:|:----------:|
|all_tickets.csv|test.py|ticket|
|emotions-train.arff|test_emotions_2.py|emotions_data|
|yeast.arff|test_yeast.py|yeast|
|scene-train.arff|test_scene.py|scene|
|flags-train.arff|test_flags.py|flags|
|birds-train.arff|test_birds.py|birds|
