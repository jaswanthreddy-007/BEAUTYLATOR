import csv 

def searchingItems(file,searching,ls): 
	for data in file: 
		print(data[0])
def excelReader(): 
	with open('sheet2.csv', encoding='utf8') as csv_file:
		file = csv.reader(csv_file)
		ls = []
		mapping = {}
		listCount = 0
		for data in file:
			if listCount != 0:
				ls.append(data[2])
				# moving through the file and appending the values in the dictionary
				mapping[listCount] = [data[0], data[2], data[3]]
			listCount += 1
		ls.sort(reverse=1)  # reversing the list to get the top list
		topTen = {}    # dictionary with top 5 members
		topvalue = 0
		count = 0
		for num in ls:
			if count >= 5:
				break
			searching = num
			for key, value in mapping.items():
				if searching == value[1]:
					# value[0] == name , value[1]=face percentage , value[2]=photo path
					topTen[topvalue] = [value[0], value[1], value[2]]   #NAME   AVERAGE    PATH 
					topvalue += 1
			count += 1
		print(topTen)

		return topTen


#x=excelReader()

			