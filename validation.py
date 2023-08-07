def validate(username,password,request):
	if username == 'user1' and password == 'user1':
		request.session["uid"] = "user1"
		return "user"
	elif username == 'user2' and password == 'user2':
		request.session["uid"] = "user2"
		return "user"
	elif username == 'user3' and password == 'user3':
		request.session["uid"] = "user3"
		return "user"
	elif username == 'user4' and password == 'user4':
		request.session["uid"] = "user4"
		return "user"
	elif username == 'user5' and password == 'user5':
		request.session["uid"] = "user5"
		return "user"
	else:
		return "invalid"