import wget
url = {'https://storage.googleapis.com/kaggle-data-sets/17839/23942/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20210207%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210207T011612Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=71a9341bb334fdf815da79981789519fb14acc272de7e1510fc014f722682293a2713e706c3047c97904342969b32cbd85af23aa92011219242fee1b4d542edcf80517309422d028bb4c9027113cb5f4eaed2df617f6b3fcb279a8e0ccb93b53721d6f84079827db658acc39998e814c1cfb96f0ff394e4806fd911d48a034705997c2129f9af47f57d716711763e1f5366be472153cd90d1341456a56b9b0da5778a596ef5ce2e38c9c06f0b8f663544af23469dd3e13004fe7a912c74ec98e804a1138489617970333cab2f8705dff29be3b7964c34e44b2716fba6f0a6ade0cdcd6873fdbeca24f4b55c2bc708537831190103fe18fe13afe67224134d466'}
wget.download(url, 'OCT2017/test.zip')