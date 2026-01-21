cd /home/dpinterview/web
sudo /home/linlab/cpp_venv/bin/gunicorn -w 10 --timeout 120 --bind 127.0.0.1:45000 wsgi:app
