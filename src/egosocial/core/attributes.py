# !/usr/bin/env python
# -*- coding: utf-8 -*-

class AttributeSelector:
    def __init__(self, all_attrs):
        body_attributes = self.filter_by_keyword(all_attrs, 'body')
        face_attributes = self.filter_by_keyword(all_attrs, 'face')
        face_attributes.extend(self.filter_by_keyword(all_attrs, 'head'))

        self._selector = {'all': all_attrs,
                          'body': body_attributes,
                          'face': face_attributes}

    def filter(self, query):
        query = query.lower()
        if query in self._selector:
            selected_attributes = self._selector[query]
        else:
            selected_attributes = self.filter_by_keyword(self._selector['all'],
                                                         query)

        return selected_attributes

    def filter_by_keyword(self, attribute_list, key):
        return [attr_name for attr_name in attribute_list if key in attr_name]
