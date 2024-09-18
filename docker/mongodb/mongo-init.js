print('Start creating database ##########################')
db = db.getSiblingDB('zjz');
db.createUser(
    {
        user: 'zjz_user',
        pwd:  'qs123456..',
        roles: [{role: 'readWrite', db: 'zjz'}],
    }
);
print('End creating database ##########################')