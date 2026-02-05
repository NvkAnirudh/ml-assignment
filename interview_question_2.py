# Write a class Validator which takes a string and then chain method

 
# Example use age: Validator(“Anirudh”).minLength(3).maxLength(10).validate()

class Validator:
    def __init__(self, value):
        self.value = value
        self.is_valid = True
    
    def required(self):
        if self.is_valid:
            if self.value is None:
                self.is_valid = False
            elif isinstance(self.value, str) and len(self.value) == 0 and not self.value.strip():
                self.is_valid = False
        return self
    
    def is_string(self):
        if self.is_valid:
            self.is_valid = isinstance(self.value, str)
        return self
    
    def minLength(self, min_len):
        if self.is_valid: 
            self.is_valid = len(self.value) >= min_len
        return self
    def maxLength(self, max_len):
        if self.is_valid: 
            self.is_valid = len(self.value) <= max_len
        return self
    def validate(self):
        return self.is_valid
        
val = Validator("Anirudh")\
.required()\
.is_string()\
.minLength(3)\
.maxLength(10)\
.validate()

print(val)
    