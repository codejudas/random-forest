Email Spam Dataset

1. Source
	-	Creators: 
		Mark Hopkins, Erik Reeber, George Forman, Jaap Suermondt 
		Hewlett-Packard Labs, 1501 Page Mill Rd., Palo Alto, CA 94304 

	-	Donor: 
		George Forman (gforman at nospam hpl.hp.com) 650-857-7835

2. Sizes of dataset
	- 	Training Set: 3601 data points
	-	Validation Set: 500 data points
	- 	Test Set: 500 data points

3. Features
	- 	48 continuous real [0,100] attributes of type word_freq_WORD
		percentage of words in the e-mail that match WORD, i.e. 100 * (number of times the WORD appears in the e-mail) / total number of words in e-mail. A "word" in this case is any string of alphanumeric characters bounded by non-alphanumeric characters or end-of-string.

	-	6 continuous real [0,100] attributes of type char_freq_CHAR
		= percentage of characters in the e-mail that match CHAR, i.e. 100 * (number of CHAR occurences) / total characters in e-mail

	-	1 continuous real [1,...] attribute of type capital_run_length_average
		= average length of uninterrupted sequences of capital letters

	-	1 continuous integer [1,...] attribute of type capital_run_length_longest
		= length of longest uninterrupted sequence of capital letters

	-	1 continuous integer [1,...] attribute of type capital_run_length_total
		= sum of length of uninterrupted sequences of capital letters
		= total number of capital letters in the e-mail

4. Format
	-	In each *Labels.csv, each line has one label, either 0 or 1. 1 represents an instance of spam while 0 represents a non-spam email.
	-	In each *Features.csv, each line represents a feature vector where each feature is separated by a comma.
