all:
	ebook-convert flashcards.html carnd.epub --page-breaks-before="//*[name()='h1' or name()='h2' or name()='dd' or name()='dt']"
