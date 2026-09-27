# Notes on Formal Presentation

*English translation of `äußerliche_Form.md` (formal and content guidelines issued by the university for the thesis).*

## Notes on Formal Presentation

The outermost cover page is prescribed and fixed. You can download it from the intranet once your registration has been accepted.

Directly behind it there must follow a declaration, signed by you, stating that you wrote the thesis independently and used no sources or aids other than those you have cited. After that, provide a summary (abstract) in both German and English, each on a page of its own.

In your well-structured table of contents, make sure that every entry is followed by enough "substance" (that is, at least roughly one page of text). Subdividing too finely ("4.3.6.1") is not helpful. In our experience, subdivision down to the first sublevel ("4.3") is almost always sufficient.

You will frequently refer to information that is already available elsewhere. The literature you use must be listed at the end in a bibliography. It has proven effective to number all entries in square brackets ("[1], [2], …" or "[1, 2]"). In the running text, you then use a reference such as "[3]" like a word in its own right, simply inserting it at a suitable place in the flowing text (without "see" and the like), for example:

> "Karlsruhe University of Applied Sciences [1] offers degree programmes in computer science [2] and business information systems [3] and is one of the largest universities of applied sciences within Baden-Württemberg [4, 5, 9]."

Always keep in mind the declaration you have submitted (see above): taking over individual lines of text (including ones you translated yourself), entire passages, tables, images and so on from a book, a journal, the internet and so forth without citing the source constitutes plagiarism and therefore leads to your thesis not being accepted.

Regarding the use of AI tools such as ChatGPT, please observe the separate information sheet *Umgang mit KI-Medien* (Dealing with AI Media).

Proper names (Apple, and so on) should be highlighted (typically in italics), and on first use you should point to the corresponding literature or website.

Number figures and tables separately from one another, consecutively and divided by chapter, that is, in Chapter 1 you assign the numbers 1.1, 1.2, and so on. Always provide a short, summarising caption for every figure and every table. Please discuss with your supervisor whether a list of figures or a list of tables at the end of the thesis offers any real added value.

An optional appendix can hold extensive explanations, data sheets, user manuals, source code and the like.

The page numbers should correspond to the actual page numbers in the PDF document. It is common to use Roman numerals for the first pages, before the first chapter.

A LaTeX tip: if you write a word such as "Client-System", LaTeX interprets the single minus sign as a hyphenation break, which you essentially never need, because LaTeX hyphenates automatically. For the hyphen you actually want, write "Client--System" instead, that is, two minus signs. A long dash, like the one here, is produced by three minus signs.

Another LaTeX tip: sometimes LaTeX hyphenates incorrectly, especially with English terms while German hyphenation is active. You can, however, specify the desired hyphenation points using the otherwise invisible character combination "\-", that is, write for example "Sil\-ben\-tren\-nung".

Spelling mistakes, lines of text running past the right-hand margin, blurry images, illegible labels and so on show that an author did not consider it necessary to go through their work carefully one last time and submit it in visually excellent condition. Naturally, that will not happen to you.

## Notes on the Content

The following general notes are helpful for producing work of high quality in terms of content. You may of course also deviate from them, since every thesis has its own particular content and constraints. The suggested chapter headings are therefore only example working titles.

Please put yourself concretely in the position of an external reader who is interested in your topic and has roughly the same prior knowledge of computer science as you do (that is, they can program, work with databases, are familiar with distributed systems, and so on). However, they are not yet a specialist in your subject area (which is precisely why they are interested in your work), and they also do not know your company, its projects and departments, and so on. It is exactly for this audience that your thesis should be written in an engaging and comprehensible way, offering appropriate added value for the reading effort they invest.

What matters is not producing as many pages as possible. Would you feel like reading 150 pages if 70 would have done the job? On the contrary, your thesis should get your idea across briefly, crisply and to the point. With numerous comprehensible examples, screenshots of the resulting application and so on, your thesis will become substantial enough anyway, so do not worry about that.

Likewise, you do not need to mention constantly that this is a final thesis or even an examination requirement. That is completely irrelevant to the external reader interested in the content. Instead, simply speak of "the present work".

### Introduction

Typically your first chapter has four sections. For a successful start, begin with a brief motivation. For example, describe a typical problem in your subject area that everyone has heard of or, better still, that everyone has "suffered" from at some point. That way the reader is certain to be interested and will read on.

Then state the goals and also the achievements of your work. What new insights does the reader gain as added value?

Give a brief overview of the environment of your work and briefly introduce the company, its business areas, and also the department or project in which you are producing the thesis.

Finally, present the structure of your thesis (beginning with the second chapter). Give two or three short sentences on the content of each chapter. Later on, too, you should begin each chapter with a short summary.

### State of the Art

Here you give an overview of previous or similar solutions and systems and point out their shortcomings. In doing so, you once again motivate the goals of your work in detail. In particular, justify why it is necessary to do something new at all. Delimit your work from systems that already exist or that are to be realised by others. Are there fixed constraints?

### System Design

Present your conceptual solution (for example the hardware and/or software design). At this point it is about the fundamental ideas, not about implementation details. Be sure to justify your decisions. Alternatives should be contrasted and their advantages and disadvantages assessed.

### Implementation

In this chapter you address the realisation of your idea. Avoid explaining every tiny detail at excessive length. Instead, build sensibly on the reader's prior knowledge. A specific algorithm can, for example, also be described in terms of its content, without filling pages of paper with lengthy listings (these belong in the appendix, if anywhere).

Use meaningful examples to show where you implemented something particularly "cleverly", but also where unforeseen difficulties arose, so that the reader does not later run into the same problems again. Lay out the solution paths you chose. Alternatives can be contrasted at the implementation level as well. Also give an overview of the languages and tools you used.

### Results

In this final chapter you do a bit of "advertising" for your work. Use suitable examples to present the progress you achieved. Can it be measured concretely (for example through shorter execution times)? Contrast the previous state and the now (hopefully) better state in detail.

While working, you will often come across aspects that could be improved further but for which you lack the time, for instance. Name these further steps in a closing outlook. In general, you are free to develop your own ideas anywhere. Could the entire task perhaps have been accomplished quite differently or more easily by another route (one your company did not even think of)? Show initiative and write your ideas down!

Even if you now have the impression that there is a lot to keep in mind, many things will later come quite easily, and writing your own thesis is also a lot of fun and satisfying. And in the end it really is your work. The university does, on the basis of the higher education act in conjunction with the study and examination regulations, have a claim to the original of the final thesis, since under higher education law it counts as an examination requirement. As the author, however, you fundamentally hold the sole copyright in your thesis and the resulting exploitation and usage rights.
