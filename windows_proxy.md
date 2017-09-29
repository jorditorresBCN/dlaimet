# How to solve Windows connectivity issues
If you can not access to your Docker container using your browser, try this:

1. Open a **cmd-terminal** or **PowerShell** on Windows 10 Pro or the Docker CLI on other Windows editions. Use this command and save the IP:
```
docker-machine ip default
```

2. Open a **cmd-terminal** or **PowerShell** with administrator privileges (Right click + Open as Administrator), you can search for it using the search bar in the start menu.

3. You will have to run the following command for each port that you need to open:
```
netsh interface portproxy add v4tov4 listenaddress=127.0.0.1 listenport=PORT connectaddress=IP connectport=PORT
```

##### In our project we will need the ports 8888 and 6006, assuming your IP is 192.168.99.100, run:

```
netsh interface portproxy add v4tov4 listenaddress=127.0.0.1 listenport=8888 connectaddress=192.168.99.100 connectport=8888
netsh interface portproxy add v4tov4 listenaddress=127.0.0.1 listenport=6006 connectaddress=192.168.99.100 connectport=6006
```
