from django.db import models

# Create your models here.
class BugPrediction(models.Model):
    product_name=models.CharField(max_length=450)
    component_name=models.CharField(max_length=450)
    short_desc=models.CharField(max_length=450)
    status_category=models.CharField(max_length=100)
    status_code=models.CharField(max_length=100)
    bug_fix=models.CharField(max_length=100)
    severity_code=models.CharField(max_length=100)
    bug_severity=models.CharField(max_length=100)