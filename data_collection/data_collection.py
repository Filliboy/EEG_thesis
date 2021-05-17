import keyboard
import pickle
import time

from .window_class import WindowMgr


def main(subject, num_songs,spotify_handle):
    i=0
    w=WindowMgr()
    like_list=[]
    val_list=[]
    arous_list=[]
    song_list=[]
    valid_list=[]
    familiarity_list=[]
    
    while i<num_songs:

        while True:
            w.find_window_wildcard(".*EmotivPRO 2.6.*")#Switch to emotivPRO
            w.move_window(550,0,750,620)#Split-screen EP and WPS
            w.set_foreground()
            keyboard.press_and_release('1') #Add start marker
            time.sleep(0.5)

            w.find_window_wildcard(".*Windows PowerShell.*")
            w.move_window(-7,0,570,620)#Split-screen EP and WPS 
            
            w.set_handle(spotify_handle) #Switch to spotify
            w.Maximize()
            w.set_foreground()

            if i>0:
                keyboard.press_and_release('ctrl+shift+down')#Unmute

            keyboard.press_and_release('shift+right')
            time.sleep(0.75) 
            keyboard.press_and_release('shift+right')
            time.sleep(0.75) 
            keyboard.press_and_release('shift+right')
            time.sleep(0.75) 
            keyboard.press_and_release('shift+right') #wind 60 seconds
            time.sleep(0.25) #3sec pretrail baseline
            keyboard.press_and_release('space') #play song
            print("Playing music...\nIf no music is playing, please wait 30 seconds and press n to replay song.")
            time.sleep(3)
            song_list.append(str(i)+'. '+w.get_window_name()) #Add song name

            w.find_window_wildcard(".*Windows PowerShell.*") #Split-screen EP and WPS
            w.set_foreground()
             
            w.find_window_wildcard(".*EmotivPRO 2.6.*") #Split-screen EP and WPS 
            w.set_foreground()

            time.sleep(27) #Play for 30 seconds (27)

            keyboard.press_and_release('2') #Add end marker
            time.sleep(0.5)

            
            w.set_handle(spotify_handle) #Switch to spotify
            w.set_foreground()
        
            keyboard.press_and_release('ctrl+shift+down') #mute
            time.sleep(0.5)
            keyboard.press_and_release('ctrl+right') #change song
            
            if i <(num_songs-1):
                time.sleep(0.3)
                keyboard.press_and_release('space') #pause


            time.sleep(0.2)
            w.find_window_wildcard(".*EmotivPRO 2.6.*") #Switch to emotivPRO
            w.move_window(550,0,750,620)
            w.set_foreground()

            time.sleep(0.2)
            w.find_window_wildcard(".*Windows PowerShell.*")
            w.set_foreground() #shift to powershell

            #Check for successful test    
            while True:
                answer = input('Was the test successful?[y/n]')
                if answer== 'y' or answer=='yes' or answer=='n' or answer=='no':
                    break
                else:
                    print("Invalid input. Try again.")

            if answer =='y' or answer =='yes':
                valid_list.append('y')
                break
            else:
                song_list.pop()
                valid_list.append('n')

                w.find_window_wildcard(".*EmotivPRO 2.6.*")#move to emotivPRO
                w.set_foreground()
                keyboard.press_and_release('3') #Add unvalid sample marker
                time.sleep(1)

                w.set_handle(spotify_handle)
                w.Maximize()
                w.set_foreground()
                time.sleep(0.7)
                keyboard.press_and_release('ctrl+left')#Replay last song (switch back)
                time.sleep(0.3)
                keyboard.press_and_release('space')#pause


                w.find_window_wildcard(".*EmotivPRO 2.6.*") #Split-screen EP and WPS
                w.set_foreground()

                w.find_window_wildcard(".*Windows PowerShell.*") #Split-screen EP and WPS
                w.set_foreground()
                print("Unsuccessfull test")
                input('Ready to replay the song? Press enter to continue')

        #-------------------------Self labeling: add user inputs--------------------
        while True:
            #Rate liking
            answer=input('Enter rating for liking (between 1 and 10): ')
            if answer.isdigit():
                rate=int(answer)
                if rate>=1 and rate<=10:
                    like_list.append(rate)
                    print('You answered: {0} on song #{1}'.format(rate,i))
                    print()
                    break
            print('Not a number or outside range, try again: enter number between 1 and 10: ')

        while True:
            #Rate valance
            answer=input('Enter rating for valance (between 1 and 10): ')
            if answer.isdigit():
                rate=int(answer)
                if rate>=1 and rate<=10:
                    val_list.append(rate)
                    print('You answered: {0} on song #{1}'.format(rate,i))
                    print()
                    break
            print('Not a number or outside range, try again: enter number between 1 and 10: ')

        while True:
            #Rate arousal
            answer=input('Enter rating for arousal (between 1 and 10): ')
            if answer.isdigit():
                rate=int(answer)
                if rate>=1 and rate<=10:
                    arous_list.append(rate)
                    print('You answered: {0} on song #{1}'.format(rate,i))
                    print()
                    break
            print('Not a number or outside range, try again: enter number between 1 and 10: ')

        while True:
            #Rate familiarity
            answer=input('Enter rating for familiarity (between 1 and 10): ')
            if answer.isdigit():
                rate=int(answer)
                if rate>=1 and rate<=10:
                    familiarity_list.append(rate)
                    print('You answered: {0} on song #{1}'.format(rate,i))
                    print()
                    break
            print('Not a number or outside range, try again: enter number between 1 and 10: ')
    
        #Save ratings 

        rating_song_dict={'Likeability':like_list,'Valance':val_list, 'Arousal': arous_list,
        'Song':song_list, 'Familiarity':familiarity_list, 'Valid test':valid_list}
        print(rating_song_dict)
     
        f = open('ratings/ratings_s'+str(subject)+'.pkl',"wb")
        pickle.dump(rating_song_dict,f)
        f.close()

        i+=1
        if i%int(num_songs/2)==0 and i!=num_songs:
            input('HALF WAY DONE. TAKE A BREAK. PRESS ENTER TO CONTINUE. ')
        if i==num_songs:
            print('That was the last song. Well done!')
        else: 
            input('Ready for next song? Press enter to continue.')



if __name__ =='__main__':
    
    while True:
        answer=input('Enter subject id (positive number): ')#Enter subject id 
        if answer.isdigit():
            subject=int(answer)
            if subject>=0:
                break
        print('Not a number or not positive. Try again. Enter a positive number: ')

    while True:
        answer=input('Enter the number of songs(positive number): ')#Enter number of songs
        if answer.isdigit():
            num_songs=int(answer)
            if num_songs>=0:
                break
        print('Not a number or not positive. Try again. Enter a positive number: ')

    w=WindowMgr()
    w.find_window_wildcard(".*Spotify Premium.*")
    spotify_handle=w._handle
    main(subject, num_songs,spotify_handle)

    