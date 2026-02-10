# NOTA:
## Al final promocioné la materia entonces este resumen está incompleto. Perdon!


# Papers que se toman:
- [No Silver Bullet](#No-Silver-Bullet)
- [Programming As Theory Building](#Programming-As-Theory-Building)
- [The Early History Of Smalltalk](#The-Early-History-Of-Smalltalk)
- [What Is Software Design](#What-Is-Software-Design)


# No Silver Bullet
- [Link al paper](https://worrydream.com/refs/Brooks_1986_-_No_Silver_Bullet.pdf)
<br><br>
All software construction involves essential tasks. This is why learning software is so difficult, we cannot simplify the models so as to reduce complexity while still keeping the essence, as the complexity IS the essence. 
<br>
There is no single development which by its own provides an order of magnitude improvement in the complexity of software production. Thus, we cannot expect twofold gains (unlike in hardware production with transistors, electronics, etc).
<br><br>

## Inherent properties of the irreducible essence of modern software

### 1. Complexity:
No two parts are alike. Upscaling software is not as linear as doubling its size, but rather increase the different number of elements and the relations nonlinearly
### 2. Conformity:
In software there is no fact or ground rule to reach, our field of work is entirely human made (and more importantly, arbitrary)
### 3. Changeability:
Software is under constant pressure of change for already manufactured products. <br>
The idea behind pressuring software to change is that it is purely thought-stuff, and that a good software must be capable of being modified as users will take it beyond its current domain. Good software also out-lives the machine it was meant for, so it must adapt to new "vehicles" (displays, disks, computers, etc)
### 4. Invisibility:
Software is invisible and unvisualizable, attempting to put in a geometric abstraction what software represents results in non heriarchical graphs with a rather large number of possible representations of the relation we are trying to graphicate. (flow of control, the flow of data, patterns of dependency, time sequence, name-space relationships)  

## Past breakthroughs

We can see a few past major breakthroughs that have made software production easier, however they all solve accidental difficulties.

- High level languages: The most powerful one so far, freeing the user of the accidental complexity of programming with a machine. 
- Time-Sharing: No se muy bien que significa.. pero preserva immediacy  y nos permite tener un overview de la complejidad.
- Unified programmin environments: providing integrated libraries, unifying file formats 

## Potential silver bullets (y sus defectos)

### 1. High level languages advances:
TODO

### 2. Object-oriented programming:
Remove all the complexity of the expression of the design but still has the complexity of the design itself.

### 3. Artifical Intelligence: 
_"The hard thig about building software is what to say, not how to say it"_ es posiblemente la frase mas basada que vi en mi vida sobre la IA.

### 4. Expert systems:
Sort of AI assistant for debugging or givings tips, but will probably be put to use as a helper for inexperienced programmers. 

### 5. Automatic programming:

### 6. Graphical programming:

### 7. Program verification:

### 8. Environments and tools: 

### 9. Workstations:

## Promising attacks on the conceptual essence

### 1. Buy versus build
The cost of software programs can be less than a programmers yearly salary. 
Can programs be utilized freshly off the package? What adaptations are required? Having non-technical employees with machine operating capabilities seems to be the way forward. (Assuming the right software is used)

### 2. Rapid prototyping
Make fast prototypes to present the clients, it's the easiest way to avoid miscommunication.
Clients are uncapable of describing and specifying with total accuracy what they want, thus showing them results and moving forward based on that is far better

### 3. Grow, don't build
Leave behind the metaphor of building dead man-made structures and grow a complex and living software with each iteration

### 4. Great designers
Good designers build better, simpler and more efficient software and are key for the software industry, developing our minds is fundamental.

# Programming As Theory Building
- [Link al paper](https://www.dropbox.com/scl/fi/plsz64xwt399h9g8skdpy/Programming-as-Theory-Building-1.doc?rlkey=h63lekenyxgacqccyi0og8gx4&e=1&dl=0)
## Introduction
This is a paper on what programming is, but Naur regards programming as the whole activity of design and implementation.
The programmer should have a foundational knowledge build-up and use the documentation as a secondary source of information.
## Scenarios
### 1.  
He gives out an example regarding two teams, _A_ and _B_, _A_ built a software _L_ and _B_ has to build an extension (_L+M_), for that team _B_ works closely with team _A_ by asking for written documentation, annotated personal texts and the offer for *personal advice*. When team B proposed several suggestions for the extension of language L the other team found that it generally disregarded the already built in features of the language and didn't take full advantage of the model. Years go by and team B is left to continue expanding the _L+M_ language, after 10 years we can see the robust original core made by team _A_ rendered completely ineffective by a addition of several poorly added features.<br>
_(Basically, team A understands the program and thus finds logical and quick solutions to problems of their own codebase, while team B even when reading the documentation only has a superficial understanding)_   
### 2. 
A company that installs a software for live monitoring adapting the code for the particular needs has 2 types of programmers:
- Those who handle the installation and fault finding which have worked with the code for years *and from the time the system was under design*, when working on the installation they rely on their knowledge and rarely the written documentation.
- Those who work the program and are given documentation of the code as well as guidance from the staff, they regularly require assistance and are put through to the original working team, who can fix the issue with ease.
_(Basically, with large programs the adaptation, modification and correction of errors is dependant of those who have deep roots with the software itself and knowledge of the theory behind it)_

## Ryle's Notion of Theory
A person who understands the theory is someone who can give explanations, justifications and answer queries regarding the things they do

Regardin programming, having the theory implies the following:
1. Ability to explain the correlation between the solution and the affairs of the world that it helps to handle, basically giving the 1:1 between program and real life scenario.
2. Ability to justify why the program is what it is, the choice of principles and rules must remain a matter of the programmers knowledge.
3. Ability to respond to any matter of modification to the program 

## Modification vs production cost

Modifying programs is generally deemed cheaper than producing a new program altogether, but this is usually not the case.
First, no analogy can be used with man-made constructions as the program's core complexity is the theory behind it. Adding flexibility to a program is also something that may be seen as a solution, but it comes at a great cost and requires building a functionality that is only useful in the future. 
Theory cannot be modified, only the program

## Program life, death and revival
A program's theory cannot be expressed.
- Life a programmer team possessing its theory remains in control of the program and the modifications.
- Death is marked by the dissolvance of the programming team that constructed it, but it can only be noticed once modifications are required and no one can answer to those modifications intelligently.
- Revival of a program is when a new team attempts to rebuild the theory. Ideally, the new team should discard the program's text and attempt to solve it from scratch, this can lead to lower costs than trying to simply "patch" over the old structure.
- The program's life can be extended when new team members are incorporated, but it is not sufficient for them to read the written documentation and program's text but also be in close contact and working with the "founding" team.
The best way to learn is to perform activities with close supervision and guidance, in the case of programming the activity should include discussions of the relation between the program and the relevant aspects and activities of the real world and of the limits set on the real world matters dealt with by the program.
<br><br>
**Similar problems are likely to arise even when a program is kept continuously alive by an evolving team of programmers**

# The Early History Of Smalltalk
- [Link al paper](https://www.dropbox.com/scl/fi/5t8ouybcqzjrcekf32woj/THE-EARLY-HISTORY-OF-SMALLTALK.pdf?rlkey=iu9gnbfmamnzy22auglayec75&e=1&dl=0)
<br>

## What Is Software Design
- [Link al paper](https://www.developerdotstar.com/mag/articles/PDF/DevDotStar_Reeves_CodeAsDesign.pdf)
<br>