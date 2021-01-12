FROM dealii/dealii:master-focal

LABEL maintainer="luca.heltai@gmail.com"

USER $USER

RUN mkdir -p /home/$USER/.vscode-server/extensions \
    /home/$USER/.vscode-server-insiders/extensions \
    && chown -R $USER \
    /home/$USER/.vscode-server \
    /home/$USER/.vscode-server-insiders

WORKDIR $HOME