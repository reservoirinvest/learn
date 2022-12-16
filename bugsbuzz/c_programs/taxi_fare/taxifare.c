#include <stdio.h>
#include <stdlib.h>

//Function prototype
float surcharge(int dayType, int timeofDay);
float basicFare(int distance);

int main()
{
    //Input variables
    int dayType, boardHour, boardMin, distance, boardTime, timeofDay;
    float meteredFare, finalFare;

    printf("Day Type: (0 for Weekends and PH | 1 for Others) \n");
    scanf("%d", &dayType);

    printf("Board Time in hh:mm: \n");
    scanf("%d %d", &boardHour, &boardMin);

    printf("Distance in meters: \n");
    scanf("%d", &distance);

    boardTime = boardHour*60+boardMin;
    printf("Boarding time is %d minutes\n", boardTime);

    //Calculate time of day
    if((boardTime >= 0) && (boardTime <= 5*60+59)) {
        timeofDay = 0; //Midnight charge (12:00 am - 5:59 am)
    }
    else if(((boardTime > 5*60+59) && (boardTime <=9*60+29))) {
        timeofDay = 1; //PH charge (6:00 am - 9:20 am)
    }
    else {
        timeofDay = 2; //PH charge (6:00 pm - 11:59 pm)
    }

    meteredFare = basicFare(distance);
    finalFare = meteredFare + meteredFare*surcharge(dayType, timeofDay);
    printf("Day-type and Time-of-day are %d, %d\n", dayType, timeofDay);
    printf("Surcharge percent is: %.2f\n", surcharge(dayType, timeofDay));
    printf("Final Fare is: %.2f\n", finalFare);

    return 0;
    }

    //Calculating Surcharge %ge

    float surcharge(int dayType, int timeofDay){
        switch(dayType){
        case 0: // Weekend or PH
            if(timeofDay == 0) {
                    return 0.5;
            }else if (timeofDay == 1) {
                return 0;
            }else {
                return 0.25;
            }

        case 1: // Weekdays and non-PH
            if(timeofDay == 0) {
                    return 0.5;
            }
            else if (timeofDay == 1) {
                    return 0.25;
            }
            else {
                    return 0.25;
            }
        }
        return 0;
    }


    //Calculate Basic Fare
    float basicFare(int distance){

        float meteredFare;
        float flagDown=3.4, m400=0.22, m350=0.22;

        if (distance <= 1000) {
            meteredFare = flagDown;
        }
        else if ((distance > 1000) && (distance <= 10200)) {
            meteredFare = flagDown + (int) (distance - 1000)/400*m400;
        }
        else{
            meteredFare = flagDown + (int) (distance - 1000)/400*m400 + (int) (distance-10200)/350*m350;
            }
        printf("basic fare is %f\n", meteredFare);
        return meteredFare;
        }
