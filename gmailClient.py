from classifier import EmailClassifier
from simplegmail import Gmail

gmail=Gmail(client_secret_file="client_secret.json")


emails=gmail.get_unread_inbox()

classifier = EmailClassifier()
for email in emails:
  inputemail=[email.plain]
  print("="*100)
  print(inputemail[0])
  print("="*100)
  out=classifier.classify(inputemail)

  if(out[0]==0):
    print("Not spam email")
  else:
    print("Spam email")
