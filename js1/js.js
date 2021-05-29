var loction1 = 3; 
var loction2 = 4;
var loction3 = 5;
var guess;
var hits = 0;
var guesses = 0;
var isSunk = false;
while(isSunk == false){
 guess = prompt("Ready,aim fire!(enter a nember from 0-6):");
 if (guess <0 || guess > 6){
     alert("Please enter a vaild cell number!");
 }else{
     guesses = guesses + 1;
     if (guess == loction1 || guess == loction2 || guess == loction3){
         alert("HIT!");
         hits = hits + 1;
         if(hits == 3){
             isSunk =true;
             alert("You sahk my battleship!'");
         } 
         } else {
            alert("MISS");
        }
    }

}
var stats = "You took" + guesses +" guesses to sink the battleship," +
"which means your shooting accuracy was" + (3/guesses);
alert (stats) ;