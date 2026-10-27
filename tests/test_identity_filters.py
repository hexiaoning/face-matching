from datetime import date
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from face_matching.database import FaceDatabase, FaceSampleInput
from face_matching.identity_filters import PersonFilter, parse_identity
from face_matching.matching import GalleryMatcher
from face_matching.config import AppConfig
from face_matching.gui.main_window import MainWindow


def identity(body):
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    return body + '10X98765432'[sum(int(a) * b for a, b in zip(body, weights)) % 11]


def add(database, name, body, vector=(1., 0.), libraries=None):
    return database.add_person(name, identity(body), [FaceSampleInput('a.png', np.array(vector), 'test', .9)],
                               library_ids=libraries)


def test_identity_checksum_birthday_sex_future_dates_and_invalid_numbers():
    value = identity('11010120001027001')
    details = parse_identity(value, date(2026, 10, 27))
    assert details.sex == '男'
    assert details.region_code == '110101'
    assert details.age_on(date(2026, 10, 26)) == 25
    assert details.age_on(date(2026, 10, 27)) == 26
    assert parse_identity(identity('11010120001027002')).sex == '女'
    for bad in (value[:-1] + ('1' if value[-1] != '1' else '0'), 'ID-001', '110101900101001',
                identity('11010120000230001'), identity('11010120990101001'), identity('11010120000101000')):
        assert parse_identity(bad) is None
    assert parse_identity('11010519491231002x').sex == '女'


def test_filters_combine_and_invalid_ids_only_excluded_with_identity_constraints():
    number = identity('11010120001027001')
    today = date(2026, 10, 27)
    assert PersonFilter(library_id='xx', sex='男', region_prefix='1101', min_age=26, max_age=26).matches(number, ('xx',), today)
    for options in ({'library_id': 'yy'}, {'sex': '女'}, {'region_prefix': '310101'}, {'max_age': 25}, {'min_age': 27}):
        assert not PersonFilter(**options).matches(number, ('xx',), today)
    assert PersonFilter().matches('legacy', ())
    assert PersonFilter(library_id='xx').matches('legacy', ('xx',))
    assert not PersonFilter(sex='男').matches('legacy', ('xx',))
    with pytest.raises(ValueError):
        PersonFilter(min_age=40, max_age=30)
    with pytest.raises(ValueError):
        PersonFilter(region_prefix='11010')


def test_library_membership_persistence_and_filter_restrict_actual_matching(tmp_path):
    database = FaceDatabase(tmp_path / 'faces.db')
    xx, yy = database.create_library('xx库'), database.create_library('yy库')
    male = add(database, '甲', '11010120000101001', libraries=[xx, yy])
    female = add(database, '乙', '31010119990101002', vector=(.95, .05), libraries=[yy])
    matcher = GalleryMatcher(database, 'test', threshold=.4, min_margin=0)
    assert matcher.person_count == 2
    matcher.person_filter = PersonFilter(library_id=yy, sex='女', region_prefix='31')
    matcher.refresh()
    assert matcher.person_count == 1
    assert matcher.match(np.array([1, 0])).person_id == female
    matcher.person_filter = PersonFilter(library_id=xx, sex='女')
    matcher.refresh()
    assert matcher.person_count == 0
    assert not matcher.match(np.array([1, 0])).accepted
    database.set_person_libraries(male, [yy])
    reopened = FaceDatabase(database.path)
    assert reopened.get_person(male).library_ids == (yy,)
    assert 'default' not in reopened.get_person(male).library_ids
    reopened.rename_library(yy, '新名称')
    assert reopened.get_person(male).library_names == ('新名称',)
    reopened.delete_library(yy)
    assert reopened.get_person(male).library_ids == ('default',)
    assert reopened.get_person(female).photo_count == 1
    with pytest.raises(ValueError):
        database.delete_library('default')


def test_upgrade_old_database_and_failed_membership_update_is_atomic(tmp_path):
    path = tmp_path / 'faces.db'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE persons(id TEXT PRIMARY KEY, name TEXT NOT NULL, id_card TEXT UNIQUE NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)')
        connection.execute("INSERT INTO persons VALUES('old', '原人员', '旧证号', '2020', '2020')")
    database = FaceDatabase(path)
    assert database.get_person('old').library_ids == ('default',)
    xx = database.create_library('xx')
    database.set_person_libraries('old', [xx])
    with pytest.raises(sqlite3.IntegrityError):
        database.set_person_libraries('old', ['missing'])
    assert database.get_person('old').library_ids == (xx,)
    assert FaceDatabase(path).get_person('old').library_ids == (xx,)


def test_ui_filters_gallery_and_people_without_losing_library_photos(qt_runtime, tmp_path, monkeypatch):
    monkeypatch.setenv('FACE_MATCHING_HOME', str(tmp_path))
    window = MainWindow(AppConfig())
    try:
        xx = window.database.create_library('xx库')
        female = add(window.database, '甲', '11010120000101002', libraries=[xx])
        add(window.database, '乙', '31010119990101001')
        matcher = GalleryMatcher(window.database, 'test')
        window.engine = SimpleNamespace(matcher=matcher, refresh_gallery=matcher.refresh)
        for controls in (window.query_filters, window.people_filters):
            controls.refresh_libraries()
            controls.library.setCurrentIndex(controls.library.findData(xx))
            controls.sex.setCurrentText('女')
        assert window.apply_query_filter()
        assert matcher.person_count == 1
        assert matcher.identities[0].person_id == female
        window.apply_people_filter()
        assert window.people_table.rowCount() == 1
        assert window.people_table.item(0, 5).text() == 'xx库'
        assert window.people_table.item(0, 6).text() == '女'
        assert len(window.database.list_stored_samples()) == 2
    finally:
        window.close()
        window.deleteLater()


def test_failed_face_initialization_keeps_complete_app_and_can_retry(qt_runtime, tmp_path, monkeypatch):
    from face_matching.errors import FaceMatchingError
    monkeypatch.setenv('FACE_MATCHING_HOME', str(tmp_path))
    window = MainWindow(AppConfig())
    errors = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: errors.append(args[-1]))
    def fail(*args):
        raise FaceMatchingError('test GPU unavailable')
    monkeypatch.setattr('face_matching.gui.main_window.assert_gpu_available', fail)
    try:
        assert not window.initialize()
        assert not window.initialize()
        assert len(errors) == 2
        assert window.tabs.count() == 7
        assert window.image_tools.open_button.isEnabled()
        assert window.video_detection.open_button.isEnabled()
        assert window.visual_search.open_button.isEnabled()
        assert window.engine is None
    finally:
        window.close()
        window.deleteLater()
