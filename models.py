from django.db import models

# Create your models here.
# class Keys(models.Model):
# 	u_id=models.IntegerField(primary_key=True)
# 	uname = models.CharField(max_length=255)
# 	p_key1 = models.CharField(max_length=255)
# 	p_key2 = models.CharField(max_length=255)

class Hashes(models.Model):
	h_id=models.IntegerField(primary_key=True)
	uname = models.CharField(max_length=255)
	hash_val = models.CharField(max_length=255)