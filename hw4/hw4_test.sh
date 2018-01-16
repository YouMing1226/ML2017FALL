wget -O word2vec https://www.dropbox.com/s/b4l3r99ua3bxrc8/word2vec?dl=1 
wget -O model.h5 https://www.dropbox.com/s/yuw3wmdvnafy72u/model.h5?dl=1
python3 train.py test --load_model model.h5 --test_path $1 --result_path $2
