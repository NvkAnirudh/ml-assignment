# Example input:
# {
#  "fullName": "Rohit Sharma",
#  "age": 29,
#  "address": {
#   "street": "MG Road",
#   "city": "Bengaluru",
#   "location": {
#      "lat": 12.9716,
#      "lng": 77.5946
#   }
#  }
# }
 
 
# Example Output:
# {
#  "fullName": "Rohit Sharma",
#  "age": 29,
#  "address.street": "MG Road",
#  "address.city": "Bengaluru",
#  "address.location.lat": 12.9716,
#  "address.location.lng": 77.5946
# }

def flatten_(input_dict, parent_key):
    flatten_dict = {}
    sep = '.'
    for k, v in input_dict.items():
        
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        
        if isinstance(v, dict):
            flatten_dict.update(flatten_(v, new_key))
        else:
            flatten_dict[new_key] = v
    return flatten_dict

input_dict = {
 "fullName": "Rohit Sharma",
 "age": 29,
 "address": {
   "street": "MG Road",
   "city": "Bengaluru",
   "location": {
     "lat": 12.9716,
     "lng": 77.5946
   }
 }
}
print(flatten_(input_dict, ''))
 