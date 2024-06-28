-- 需求: 将crud表数据复制到新建表中

-- create database crud;

CREATE TABLE `crud_current` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `country` varchar(3) NOT NULL,
  `data_source` varchar(20) NOT NULL,
  `gaid` varchar(36) NOT NULL,
  `status` varchar(10) NOT NULL,
  `lock_status` varchar(20) NOT NULL,
  `lock_start` datetime DEFAULT NULL,
  `device_json` text NOT NULL,
  `device_json_new` text,
  `created` datetime NOT NULL,
  `updated` datetime NOT NULL,
  `model` varchar(54) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `gaid_uniq_idx` (`gaid`),
  KEY `country_idx` (`country`),
  KEY `created_idx` (`created`)
) ENGINE=InnoDB AUTO_INCREMENT=826765 DEFAULT CHARSET=utf8mb4;

insert into crud_current select * from crud;
