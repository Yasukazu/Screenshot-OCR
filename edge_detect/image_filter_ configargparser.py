import configargparser as cap

parser = cap.ConfigArgumentParser()
parser.read("image_filter.ini")
parser.parse_args()#shorts="edbtfp")

print("Configs:", parser.defaults)
print("Args:   ", parser.args)